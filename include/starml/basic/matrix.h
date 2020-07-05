#pragma once
// #include "starml/basic/storage.h"
#include "starml/basic/device.h"
#include "starml/basic/allocator.h"

namespace starml {

class Matrix {
 public:
  Matrix(int row, int col, DeviceType device_type);
  float* data() const { return static_cast<float*>(data_ptr_.get()); }

 private:
  size_t size_;
  Device device_;
  Allocator* allocator_;
  DataPtr data_ptr_;
  int dims[2];
};
}  // namespace starml