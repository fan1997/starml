#pragma once
#include "starml/basic/storage.h"

namespace starml {

class Matrix {
 public:
  Matrix(int row, int col, DeviceType device_type);
  float* data() const { return storage_->data(); }

 private:
  Storage* storage_;
  int dims[2];
};
}  // namespace starml