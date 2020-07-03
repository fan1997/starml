#pragma once
#include <cstddef>
#include <unordered_map>
#include <mutex>
#include "starml/basic/device.h"
#include "starml/basic/data_ptr.h"
#include <iostream>

namespace starml {
class Allocator {
 public:
  virtual ~Allocator() = default;
  virtual void* allocate_raw(size_t num_bytes) const = 0;
  virtual void deallocate_raw(void* ptr) const; 
  virtual DeleterFnPtr raw_deleter() const = 0;
  DataPtr allocate(size_t num_bytes) const;
};

class AllocatorRegistry {
 public:
  static AllocatorRegistry* singleton(); 
  void set_allocator(DeviceType device_type, Allocator* allocator);
  Allocator* allocator(DeviceType device_type);
 private:
  AllocatorRegistry();
  AllocatorRegistry(AllocatorRegistry const&) = delete;
  AllocatorRegistry& operator=(AllocatorRegistry const&) = delete;
  std::unordered_map<int, Allocator*> allocators_;
  std::mutex mu_;
};

class AllocatorRegister {
 public:
  AllocatorRegister(DeviceType device_type, Allocator* allocator);
};

#define STARML_REGISTER_ALLOCATOR(device, allocator)                        \
  static AllocatorRegister g_allocator_register(device, allocator)          \

Allocator* get_allocator(DeviceType device_type);
}  // namespace starml