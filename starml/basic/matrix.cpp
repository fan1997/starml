#include <iostream>
#include "starml/basic/matrix.h"
#include "starml/basic/dispatch.h"
namespace starml {
int Matrix::print_limited[2] = {20, 20};
// int Matrix::print_limited[1] = 20;

Matrix::Matrix() : allocator_(NULL), data_ptr_(NULL) {}
Matrix::Matrix(int row, int col, DeviceType device_type, DataTypeKind data_type)
    : dtype_(DataType(data_type)), device_(Device(device_type)) {
  this->size_ = row * col * dtype_.size();
  this->allocator_ = get_allocator(device_type);
  this->data_ptr_ = allocator_->allocate(size_);
  this->dims[0] = row;
  this->dims[1] = col;
}
const int* Matrix::shape() const { return this->dims; }
int Matrix::rows_num() const { return this->dims[0]; }
int Matrix::cols_num() const { return this->dims[1]; }
DataTypeKind Matrix::data_type() const { return this->dtype_.type(); }

std::ostream& operator<<(std::ostream& os, const Matrix& rhs) {
  int num_of_rows = std::min(rhs.rows_num(), Matrix::print_limited[0]);
  int num_of_cols = std::min(rhs.cols_num(), Matrix::print_limited[1]);
  os << "Matrix "
     << "(" << rhs.rows_num() << ", " << rhs.cols_num << ")\n";
  STARML_DISPATCH_TYPES(rhs.data_type(), "matrix printer", [&] {
    const scalar_t* data = rhs.data<scalar_t>();
    for (int i = 0; i < num_of_rows; i++) {
      for (int j = 0; j < num_of_cols; j++) {
        os << data[i * num_of_cols + j] << " ";
      }
      os << '\n';
    }
  });
}
}  // namespace starml