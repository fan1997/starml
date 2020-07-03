#pragma once
#include "starml/basic/allocator.h"

namespace starml {
class CPUAllocator : public Allocator {
 public:
  CPUAllocator();
  ~CPUAllocator() override;
  void* allocate_raw(size_t num_bytes) const override;
  DeleterFnPtr raw_deleter() const override;
  static void delete_fn(void *ptr);
};
Allocator* cpu_allocator();
}  // namespace starml