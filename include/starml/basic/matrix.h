#pragma once
#include "starml/basic/allocator.h"
#include "starml/basic/device.h"
#include "starml/basic/type.h"

namespace starml {

class Matrix {
 public:
  Matrix();
  Matrix(int row, int col, DeviceType device_type, DataTypeKind data_type);
  template <typename T>
  T* data() const {
    // Previous judgement is needed to check whether the template datatype
    // is valid.
    if (!dtype_.is_valid<T>()) {
      return NULL;
    }
    return static_cast<T*>(data_ptr_.get());
  }
  const int *shape() const;
  int rows_num() const;
  int cols_num() const;

 private:
  size_t size_;
  Device device_;
  DataType dtype_;
  Allocator* allocator_;
  DataPtr data_ptr_;
  int dims[2];
};

}  // namespace starml
