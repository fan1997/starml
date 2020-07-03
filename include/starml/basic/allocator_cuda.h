#pragma once
#include "starml/basic/allocator.h"

namespace starml {
class CUDAAllocator : public Allocator {
 public:
  CUDAAllocator() {}
  ~CUDAAllocator() override {}
  void* allocate_raw(size_t num_bytes) const override;
  DeleterFnPtr raw_deleter() const override;
  static void delete_fn(void *ptr);
};
static CUDAAllocator g_cuda_allocator;
Allocator* cuda_allocator();

}  // namespace starml