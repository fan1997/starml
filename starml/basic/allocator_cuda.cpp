#include "starml/basic/allocator_cuda.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>

namespace starml {
void *CUDAAllocator::allocate_raw(size_t num_bytes) const {
  void *d_ptr = 0;
  cudaMalloc(&d_ptr, num_bytes);
  return d_ptr;
}

DeleterFnPtr CUDAAllocator::raw_deleter() const { return &delete_fn; }

void CUDAAllocator::delete_fn(void *ptr) { cudaFree(ptr); }

static CUDAAllocator g_cuda_allocator;
STARML_REGISTER_ALLOCATOR(DeviceType::CUDA, &g_cuda_allocator);

Allocator* cuda_allocator() {
  static Allocator* allocator = 
      AllocatorRegistry::singleton()->allocator(DeviceType::CUDA);
  return allocator;
}

}  // namespace starml