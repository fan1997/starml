#pragma once
#include <iostream>
#include <type_traits>
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
  DataType data_type() const;
  Device device_type() const;
  void* raw_data() const;
  const void* raw_mutable_data() const;
  void print(std::ostream& os = std::cout) const;
  static int print_limited[2];
  Matrix to(DeviceType device_type) const;

 private:
  size_t size_;
  Device device_;
  DataType dtype_;
  Allocator* allocator_;
  DataPtr data_ptr_;
  int dims[2];
};
std::ostream& operator<<(std::ostream& os, const Matrix& rhs);
}  // namespace starml