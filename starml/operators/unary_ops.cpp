#include "starml/operators/unary_ops.h"

namespace starml {
STARML_DEFINE_DISPATCHER(exp_dispatcher);

Matrix exp(const Matrix& matrix, bool blocking) {
  // If the input matrix is int, then the result data type should be float
  // else the data type of result should be consistent with the input
  auto result_dtype =
      (matrix.data_type().is_int()) ? kFloat : matrix.data_type().type();
  Matrix result = Matrix(matrix.dims(), matrix.device(), result_dtype);
  exp_dispatcher(matrix, result, blocking);
  return result;
}

Matrix exp(const Matrix& matrix, Matrix& result, bool blocking) {
  exp_dispatcher(matrix, result, blocking);
  return result;
}

}  // namespace starml