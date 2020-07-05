#include "starml/basic/matrix.h"

namespace starml {
  Matrix::Matrix(int row, int col, DeviceType device_type) {
    this->size_ = row * col * sizeof(float);
    this->allocator_ = get_allocator(device_type);
    this->device_ = Device(device_type);
    this->data_ptr_ = allocator_->allocate(size_);
    this->dims[0] = row;
    this->dims[1] = col;
  }
}