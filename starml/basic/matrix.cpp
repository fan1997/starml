#include "starml/basic/matrix.h"
#include "starml/basic/dispatch.h"
#include "starml/basic/copy_bytes.h"
#include "starml/operators/factories.h"

namespace starml {
int Matrix::print_limited[2] = {20, 20};

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
DataType Matrix::data_type() const { return this->dtype_; }
void Matrix::print(std::ostream& os) const { os << *this; }
Device Matrix::device_type() const { return this->device_; }
void* Matrix::raw_data() const { return this->data_ptr_.get(); }
const void* Matrix::raw_mutable_data() const { return this->data_ptr_.get(); }
Matrix Matrix::to(DeviceType new_device_type) const {
  Matrix res = empty(rows_num(), cols_num(), new_device_type, dtype_.type());
  copy_bytes(size_, raw_mutable_data(), device_type(), res.raw_data(),
             res.device_type());
  return res;
}
std::ostream& operator<<(std::ostream& os, const Matrix& rhs) {
  int num_of_rows = std::min(rhs.rows_num(), Matrix::print_limited[0]);
  int num_of_cols = std::min(rhs.cols_num(), Matrix::print_limited[1]);
  os << "Matrix "
     << "(" << rhs.rows_num() << ", " << rhs.cols_num() << ")\n";
  auto data_type = rhs.data_type().type(); 
  STARML_DISPATCH_TYPES(data_type, "matrix printer", [&] {
    const scalar_t* data = rhs.data<scalar_t>();
    for (int i = 0; i < num_of_rows; i++) {
      for (int j = 0; j < num_of_cols; j++) {
        os << data[i * num_of_cols + j] << " ";
      }
      os << '\n';
    }
  });
  return os;
}

}  // namespace starml