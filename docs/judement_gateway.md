# Experimental Judement CUDA reservation gateway

Start the standalone server with a private socket and policy directory, then opt a PyTorch process into the V2 policy before its first CUDA allocation:

```sh
install -d -m 0700 "$HOME/.cache/judement-poc"
tools/judement_cpp_poc/build/judement-server \
  --socket "$HOME/.cache/judement-poc/control.sock" \
  --policy "$HOME/.cache/judement-poc/policy-v2.bin"
```

```sh
export JUDEMENT_POLICY_MODE=mmap_v2
export JUDEMENT_POLICY_PATH="$HOME/.cache/judement-poc/policy-v2.bin"
export JUDEMENT_SOCKET="$HOME/.cache/judement-poc/control.sock"
python my_workload.py
```

```python
import torch

with torch.cuda.judement.reserve(gpu_ids=[0, 1], timeout_s=30):
    left = torch.empty(1024, device="cuda:0")
    right = torch.empty(1024, device="cuda:1")
```

`gpu_ids` are the CUDA ordinals visible inside that process. The API matches their CUDA UUIDs to the server's physical inventory, so a client's `CUDA_VISIBLE_DEVICES` order can differ from the server's order. The full requested set is granted together or the request waits and times out. Nested leases in the same PID are rejected. The context manager releases the lease on exit and warns when live PyTorch allocations remain.

This feature governs future allocations through the patched native PyTorch caching allocator only. It does not reclaim tensors, stop kernels on existing tensors, block direct `cudaMalloc` by extensions, or isolate an unmodified process. The `cudaMallocAsync` backend, globally pluggable allocators, uncached CUDA allocation, MIG, and multiple PIDs acting as one job are outside this prototype. Start the server before starting strict-mode clients. In strict mode a missing or invalid policy denies native allocations with a Judement error. With `JUDEMENT_POLICY_MODE` unset or `off`, ordinary PyTorch behavior applies.

The server must run with `CUDA_VISIBLE_DEVICES` unset and have access to the NVIDIA driver. Its socket and policy parent directories must be owned by the current user and have mode `0700`. The policy file has mode `0600`. `--synthetic-uuids` exists solely for CPU-only server tests and does not enable GPU enforcement.

The current checkout has not yet been rebuilt as PyTorch; these commands document the intended interface, not a completed live validation. See the accompanying test and benchmark reports before relying on it.
