#pragma once
#include <cstddef>
#include <unordered_map>
#include <mutex>
#include <memory>
#include "starml/basic/device.h"

namespace starml {
// Unified function pointer for free allocated memory, 
// the only input parameter is the pointer to be free.
using DeleterFnPtr = void (*)(void*);
// In order to avoid memory leak, using shared_ptr as the underground
// type for the data of Matrix.
typedef std::shared_ptr<void> DataPtr; 

// Abstract base class of all device-allocators, in order to manage
// allocate/deallocate polymorphically.
class Allocator {
 public:
  // Deconstructor should be virtual since base class has virtual functions. 
  virtual ~Allocator() = default;
  // Allocate the given `num_bytes` space on specific device, return a pointer 
  // which point to the new allocated space.
  virtual void* allocate_raw(size_t num_bytes) const = 0;
  // Define and return the function pointer which used to free the space on given device. 
  virtual DeleterFnPtr raw_deleter() const = 0;
  // Using the proper deleter function to free the space where the parameter pointer point to.
  void deallocate_raw(void* ptr) const; 
  // Using 
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