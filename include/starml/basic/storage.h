#pragma once
#include "starml/basic/allocator.h"
#include "starml/basic/device.h"
#include "starml/basic/data_ptr.h"

namespace starml {
class Storage {
 public:
  Storage(size_t size, DeviceType device_type, Allocator* allocator);
  ~Storage();
  float* data() { return static_cast<float*>(data_ptr_.get()); }

 private:
  size_t size_;
  DataPtr data_ptr_;;
  Device device_;
  Allocator* allocator_;
};
}  // namespace starml