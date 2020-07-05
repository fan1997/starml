#pragma once
#include "starml/basic/storage.h"

namespace starml {

class Matrix {
 public:
  Matrix(){};
  // ~Matrix(){delete storage_;}
  Matrix(int row, int col, DeviceType device_type);
  float* data() const { return storage_->data(); }

 private:
  Storage* storage_;   //类指针需要显示析构？？
  int dims[2];
};

}  // namespace starml
