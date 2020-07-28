#include "starml/operators/copy.h"

namespace starml {
STARML_DEFINE_DISPATCHER(copy_dispatcher);

Matrix deep_copy(const Matrix& src, const Device& new_device, void* stream) {
  Matrix result = Matrix(src.dims(), new_device, src.data_type());
  copy_dispatcher(src, result, stream);
  return result;
}
}  // namespace starml