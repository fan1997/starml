#include "starml/operators/unary_ops.h"

namespace starml {
STARML_DEFINE_DISPATCHER(exp_dispatcher);

Matrix exp(const Matrix& matrix) {
  // If the input matrix is int, then the result data type should be float
  // else the data type of result should be consistent with the input
  auto result_dtype =
      (matrix.data_type().is_int()) ? kFloat : matrix.data_type().type();
  Matrix result = Matrix(matrix.dims(), matrix.device(), result_dtype);
  exp_dispatcher(matrix, result);
  return result;
}

Matrix exp(const Matrix& matrix, Matrix& result) {
  exp_dispatcher(matrix, result);
  return result;
}

STARML_DEFINE_DISPATCHER(log_dispatcher);

Matrix log(const Matrix& matrix, bool blocking) {
  // If the input matrix is int, then the result data type should be float
  // else the data type of result should be consistent with the input
  auto result_dtype =
      (matrix.data_type().is_int()) ? kFloat : matrix.data_type().type();
  Matrix result = Matrix(matrix.dims(), matrix.device(), result_dtype);
  log_dispatcher(matrix, result, blocking);
  return result;
}

Matrix log(const Matrix& matrix, Matrix& result, bool blocking) {
  log_dispatcher(matrix, result, blocking);
  return result;
}

STARML_DEFINE_DISPATCHER(negtive_dispatcher);
Matrix negtive(const Matrix& matrix, bool blocking) {
  Matrix result = Matrix(matrix.dims(), matrix.device(),  matrix.data_type());
  negtive_dispatcher(matrix, result, blocking);
  return result;
}
Matrix negtive(const Matrix& matrix, Matrix& result, bool blocking) {
  negtive_dispatcher(matrix, result, blocking);
  return result;
}

}  // namespace starml
