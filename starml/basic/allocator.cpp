#include "starml/basic/allocator.h"
#include "starml/basic/device.h"

namespace starml {
void Allocator::deallocate_raw(void* ptr) const {
  auto deleter = raw_deleter();
  deleter(ptr);
}

DataPtr Allocator::allocate(size_t num_bytes) const {
  void* raw_ptr = allocate_raw(num_bytes);
  auto deleter = raw_deleter();
  return {raw_ptr, deleter};
}

AllocatorRegistry::AllocatorRegistry() {}

AllocatorRegistry* AllocatorRegistry::singleton() {
  static AllocatorRegistry* alloc_registry = new AllocatorRegistry();
  return alloc_registry;
}

void AllocatorRegistry::set_allocator(DeviceType device_type,
                                      Allocator* allocator) {
  std::lock_guard<std::mutex> guard(mu_);
  allocators_[static_cast<int>(device_type)] = allocator;
}

Allocator* AllocatorRegistry::allocator(DeviceType device_type) {
  return allocators_[static_cast<int>(device_type)];
}

AllocatorRegister::AllocatorRegister(DeviceType device_type,
                                     Allocator* allocator) {
  AllocatorRegistry::singleton()->set_allocator(device_type, allocator);
}

Allocator* get_allocator(DeviceType device_type) {
  Allocator *alloc = AllocatorRegistry::singleton()->allocator(device_type);
  return alloc;
}
}  // namespace starml