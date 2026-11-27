#pragma once

#include <cstddef>
#include <cstdint>

namespace c10::cuda::judement {

constexpr std::size_t kMaxDevices = 64;
constexpr std::size_t kPidCapacity = 256;
constexpr std::size_t kPolicySize = 10304;
constexpr char kMagic[8] = {'J', 'D', 'E', 'M', 'M', 'P', '0', '2'};

struct alignas(8) PolicyHeader {
  char magic[8];
  uint32_t version;
  uint32_t header_size;
  uint32_t num_devices;
  uint32_t pid_capacity;
  uint64_t sequence;
  uint64_t flags;
  int32_t writer_pid;
  uint32_t reserved;
  uint64_t server_generation;
  uint64_t writer_start_ticks;
};

struct alignas(8) DeviceEntry {
  uint8_t uuid[16];
  int32_t owner_pid;
  uint32_t lock_state;
  uint64_t reserved;
};

struct alignas(8) PidEntry {
  int32_t pid;
  uint32_t active;
  uint64_t allowed_device_mask;
  uint64_t phase_id;
  uint64_t process_start_ticks;
};

struct alignas(8) Policy {
  PolicyHeader header;
  DeviceEntry devices[kMaxDevices];
  PidEntry pids[kPidCapacity];
};

static_assert(sizeof(PolicyHeader) == 64);
static_assert(sizeof(DeviceEntry) == 32);
static_assert(sizeof(PidEntry) == 32);
static_assert(offsetof(Policy, devices) == 64);
static_assert(offsetof(Policy, pids) == 2112);
static_assert(sizeof(Policy) == kPolicySize);
static_assert(offsetof(PolicyHeader, sequence) == 24);
static_assert(offsetof(PolicyHeader, server_generation) == 48);
static_assert(offsetof(PolicyHeader, writer_start_ticks) == 56);
static_assert(offsetof(DeviceEntry, owner_pid) == 16);
static_assert(offsetof(PidEntry, allowed_device_mask) == 8);
static_assert(offsetof(PidEntry, process_start_ticks) == 24);
#if defined(__GNUC__) && defined(__linux__)
static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__);
static_assert(__atomic_always_lock_free(sizeof(uint32_t), nullptr));
static_assert(__atomic_always_lock_free(sizeof(uint64_t), nullptr));
#else
#error "Judement V2 requires Linux GCC or Clang lock-free atomics"
#endif

template <typename T>
inline T load(const T* value) {
  return __atomic_load_n(value, __ATOMIC_SEQ_CST);
}

template <typename T>
inline void store(T* value, T next) {
  __atomic_store_n(value, next, __ATOMIC_SEQ_CST);
}

} // namespace c10::cuda::judement
