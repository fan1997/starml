#include "starml/basic/matrix.h"
#include "starml/basic/allocator.h"

namespace starml {
  Matrix::Matrix(int row, int col, DeviceType device_type) {
    size_t num_bytes = row * col * sizeof(float);
    this->storage_ = new Storage(num_bytes, device_type, get_allocator(device_type));
    this->dims[0] = row;
    this->dims[1] = col;
  }
}
