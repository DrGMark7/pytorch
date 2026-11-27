r"""Experimental local GPU reservation client for the Judement allocator policy."""

import contextlib
import math
import os
import socket
import sys
import warnings
from collections.abc import Iterator, Sequence
from typing import BinaryIO

import torch


class JudementError(RuntimeError):
    r"""Raised when a Judement reservation cannot be established or released."""


def _exchange(connection: socket.socket, reader: BinaryIO, request: str, operation: str) -> list[str]:
    connection.sendall(request.encode("ascii") + b"\n")
    response = reader.readline(4097)
    if not response or len(response) > 4096 or not response.endswith(b"\n"):
        raise JudementError("Invalid or missing response from Judement server")
    try:
        fields = response.decode("ascii").strip().split()
    except UnicodeDecodeError as exc:
        raise JudementError("Invalid response from Judement server") from exc
    if len(fields) == 3 and fields[:2] == ["J2", "ERR"]:
        raise JudementError(f"Judement {operation} failed: {fields[2]}")
    if len(fields) < 3 or fields[:3] != ["J2", "OK", operation]:
        raise JudementError(f"Unexpected Judement {operation} response: {response!r}")
    return fields[3:]


@contextlib.contextmanager
def reserve(gpu_ids: Sequence[int], timeout_s: float = 30) -> Iterator[int]:
    r"""reserve(gpu_ids, timeout_s=30) -> Iterator[int]

    Reserve all requested visible CUDA devices for a code region.

    ``gpu_ids`` are this process's logical CUDA ordinals. The reservation only
    governs allocations made by participating processes using the patched
    PyTorch native caching allocator. Releasing it does not free live tensors.

    Args:
        gpu_ids (Sequence[int]): Distinct logical CUDA device ordinals to reserve.
        timeout_s (float, optional): Maximum wait for the complete device set in
          seconds. Default: ``30``.

    Yields:
        int: The server's reservation token.

    Examples::

        >>> with torch.cuda.judement.reserve([0, 1], timeout_s=30):
        ...     x = torch.empty(1024, device="cuda:0")
        ...     y = torch.empty(1024, device="cuda:1")
    """
    if os.getenv("JUDEMENT_POLICY_MODE") != "mmap_v2":
        raise JudementError("Judement reserve requires JUDEMENT_POLICY_MODE=mmap_v2")
    if not os.getenv("JUDEMENT_POLICY_PATH"):
        raise JudementError("Judement reserve requires JUDEMENT_POLICY_PATH")
    socket_path = os.getenv("JUDEMENT_SOCKET")
    if not socket_path:
        raise JudementError("Judement reserve requires JUDEMENT_SOCKET")
    if torch.version.hip:
        raise JudementError("Judement reservation currently supports CUDA only")
    if torch.cuda.get_allocator_backend() != "native":
        raise JudementError("Judement reservation requires the native CUDA allocator")
    if torch._C._cuda_getAllocator() is not None:
        raise JudementError("Judement reservation does not support a pluggable CUDA allocator")
    if os.getenv("PYTORCH_NO_CUDA_MEMORY_CACHING") == "1":
        raise JudementError("Judement reservation does not support uncached CUDA allocation")
    if isinstance(gpu_ids, (str, bytes)):
        raise ValueError("gpu_ids must be a sequence of CUDA device ordinals")
    devices = list(gpu_ids)
    if not devices or any(type(device) is not int for device in devices):
        raise ValueError("gpu_ids must contain at least one integer CUDA device ordinal")
    if len(set(devices)) != len(devices):
        raise ValueError("gpu_ids must not contain duplicate CUDA device ordinals")
    if not isinstance(timeout_s, (int, float)) or isinstance(timeout_s, bool):
        raise ValueError("timeout_s must be a finite number between 0 and 3600")
    if not math.isfinite(timeout_s) or not 0 <= timeout_s <= 3600:
        raise ValueError("timeout_s must be a finite number between 0 and 3600")
    timeout_ms = math.ceil(timeout_s * 1000)

    device_count = torch.cuda.device_count()
    if any(device < 0 or device >= device_count for device in devices):
        raise ValueError(f"gpu_ids must be in the visible CUDA device range 0..{device_count - 1}")
    if device_count > 64:
        raise JudementError("Judement protocol supports at most 64 visible CUDA devices")
    uuids = [bytes(torch.cuda.get_device_properties(i).uuid.bytes).hex() for i in range(device_count)]
    if any(len(uuid) != 32 or uuid == "0" * 32 for uuid in uuids) or len(set(uuids)) != len(uuids):
        raise JudementError("Cannot establish a unique UUID for every visible CUDA device")

    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(5)
            connection.connect(socket_path)
            with connection.makefile("rb") as reader:
                hello = _exchange(connection, reader, f"J2 HELLO {device_count} {' '.join(uuids)}", "HELLO")
                if len(hello) != 2 or not all(field.isdecimal() for field in hello):
                    raise JudementError("Invalid Judement HELLO response")
                connection.settimeout(max(5.0, timeout_s + 5.0))
                request = f"J2 ACQUIRE {timeout_ms} {len(devices)} {' '.join(map(str, devices))}"
                lease = _exchange(connection, reader, request, "LEASE")
                if len(lease) != 3 or not all(field.isdecimal() for field in lease) or lease[1] != hello[0]:
                    raise JudementError("Invalid Judement lease or server generation")
                token = int(lease[0])
                owner_pid = os.getpid()
                connection.settimeout(5)
                try:
                    yield token
                finally:
                    if os.getpid() != owner_pid:
                        raise JudementError("Judement reservation cannot be released by a forked child")
                    active_exception = sys.exc_info()[0] is not None
                    try:
                        try:
                            live = {i: torch.cuda.memory_allocated(i) for i in devices}
                            reserved = {i: torch.cuda.memory_reserved(i) for i in devices}
                            if any(live.values()):
                                warnings.warn(
                                    f"Judement releases allocation permission while live CUDA tensors remain: "
                                    f"live_allocated_bytes={live}, reserved_bytes={reserved}",
                                    stacklevel=2,
                                )
                        except Exception as exc:
                            warnings.warn(f"Judement memory diagnostics failed: {exc}", stacklevel=2)
                    finally:
                        try:
                            released = _exchange(connection, reader, f"J2 RELEASE {token}", "RELEASE")
                            if len(released) != 2 or not all(field.isdecimal() for field in released) or released[0] != hello[0]:
                                raise JudementError("Invalid Judement RELEASE response")
                        except (OSError, JudementError) as exc:
                            if active_exception:
                                warnings.warn(f"Judement release failed: {exc}", stacklevel=2)
                            else:
                                raise
    except OSError as exc:
        raise JudementError(f"Judement server unavailable: {exc}") from exc
