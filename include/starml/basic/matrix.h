#pragma once
// #include <iostream>
#include <vector>

#include "starml/basic/allocator.h"
#include "starml/basic/device.h"
#include "starml/basic/type.h"
#include "starml/utils/loguru.h"

namespace starml {
using Shape = std::vector<int>;
class Matrix {
 public:
  Matrix();
  Matrix(Shape shape, Device device, DataType data_type);
  Matrix(Shape shape, DataType data_type, Device device);
  ~Matrix() = default;
  Matrix(const Matrix& rhs) = default;
  Matrix& operator=(const Matrix& rhs) = default;

  int size() const;
  const Shape& dims() const;
  int dim(int index) const;
  int ndims() const;
  const Device& mutable_device() const;
  Device device() const;
  const DataType& data_type() const;

  void* raw_data() const;
  const void* raw_mutable_data() const;

  Matrix to(DeviceType new_device_type) const;

  // Previous judgement is needed to check whether the template datatype
  // is valid.
  template <typename T>
  T* data() const {
    STARML_CHECK(dtype_.is_valid<T>())
        << "Input template data type is not valid since the data type for "
           "matrix is "
        << dtype_.type();
    return static_cast<T*>(data_ptr_.get());
  }
  template <typename T>
  const T* mutable_data() const {
    STARML_CHECK(dtype_.is_valid<T>())
        << "Input template data type is not valid since the data type for "
           "matrix is "
        << dtype_.type();
    return static_cast<T*>(data_ptr_.get());
  }

 private:
  void initial();
  int size_;
  Device device_;
  DataType dtype_;
  Allocator* allocator_;
  DataPtr data_ptr_;
  Shape shape_;
};
}  // namespace starml