#include <stdlib.h>
#include "starml/basic/allocator_cpu.h"

namespace starml {
CPUAllocator::CPUAllocator() {}
CPUAllocator::~CPUAllocator() {}

void* CPUAllocator::allocate_raw(size_t num_bytes) const{
  void* ptr = malloc(num_bytes);
  return ptr;
}

void CPUAllocator::delete_fn(void* ptr) { free(ptr); }

DeleterFnPtr CPUAllocator::raw_deleter() const { return delete_fn; }

static CPUAllocator g_cpu_allocator;
STARML_REGISTER_ALLOCATOR(kCPU, &g_cpu_allocator);

Allocator* cpu_allocator() {
  static Allocator* allocator =
      AllocatorRegistry::singleton()->allocator(DeviceType::CPU);
  return allocator;
}

}  // namespace starml