#include "starml/basic/matrix.h"

namespace starml {
  Matrix::Matrix() : allocator_(NULL), data_ptr_(NULL) {}
  Matrix::Matrix(int row, int col, DeviceType device_type,
                 DataTypeKind data_type)
      : dtype_(DataType(data_type)), device_(Device(device_type)) {
    this->size_ = row * col * dtype_.size();
    this->allocator_ = get_allocator(device_type);
    this->data_ptr_ = allocator_->allocate(size_);
    this->dims[0] = row;
    this->dims[1] = col;
  }
  const int* Matrix::shape() const {
    return this->dims;
  }
  int Matrix::rows_num() const {
    return this->dims[0];
  }
  int Matrix::cols_num() const {
    return this->dims[1];
  }
}