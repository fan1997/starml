#include "starml/operators/unary_ops.h"

namespace starml {
STARML_DEFINE_DISPATCHER(exp_dispatcher);

Matrix exp(const Matrix& matrix) {
  Matrix result = Matrix(matrix.dims(), matrix.device(), matrix.data_type());
  exp_dispatcher(matrix, result);
  return result;
}

}  // namespace starml