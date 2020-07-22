#include "starml/operators/binary_ops.h"

namespace starml {
STARML_DEFINE_DISPATCHER(add_dispatcher);
STARML_DEFINE_DISPATCHER(sub_dispatcher);
STARML_DEFINE_DISPATCHER(mul_dispatcher);
STARML_DEFINE_DISPATCHER(div_dispatcher);

Shape broadcast(const Matrix& matrix1, const Matrix& matrix2) {
  auto shape1 = matrix1.dims();
  auto shape2 = matrix2.dims();
  int ndims1 = matrix1.ndims();
  int ndims2 = matrix2.ndims();
  // if the trailing dimension of the two matrix are equal 
  // or one of them equal to 1
  bool can_broadcast = true;
  for (int i = ndims1 - 1, j = ndims2 - 1; i >= 0 && j >= 0; i--, j--) {
    can_broadcast = can_broadcast && (shape1[i] == shape2[j] ||
                                      shape1[i] == 1 || shape2[j] == 1);
  }
  STARML_CHECK(can_broadcast) << "Operands could not be broadcast.";
  int result_dims = (ndims1 < ndims2) ? ndims2 : ndims1;
  Shape result = Shape(result_dims);
  int k = result_dims - 1;
  int i = ndims1 - 1;
  int j = ndims2 - 1;
  while (i >= 0 && j >= 0) {
    if (shape1[i] > shape2[j]) {
      result[k--] = shape1[i--];
      j--;
    } else {
      result[k--] = shape2[j--];
      i--;
    }
  }
  while (i >= 0) {
    result[k--] = shape1[i--];
  }
  while (j >= 0) {
    result[k--] = shape2[j--];
  }
  return result;
}

Matrix add(const Matrix& matrix1, const Matrix& matrix2) {
  auto shape = broadcast(matrix1, matrix2);
  Matrix result = Matrix(shape, matrix1.device(), matrix1.data_type());
  add_dispatcher(matrix1, matrix2, result);
  return result;
}
Matrix sub(const Matrix& matrix1, const Matrix& matrix2) {
  auto shape = broadcast(matrix1, matrix2);
  Matrix result = Matrix(shape, matrix1.device(), matrix1.data_type());
  sub_dispatcher(matrix1, matrix2, result);
  return result;
}
Matrix mul(const Matrix& matrix1, const Matrix& matrix2) {
  auto shape = broadcast(matrix1, matrix2);
  Matrix result = Matrix(shape, matrix1.device(), matrix1.data_type());
  mul_dispatcher(matrix1, matrix2, result);
  return result;
}
Matrix div(const Matrix& matrix1, const Matrix& matrix2) {
  auto shape = broadcast(matrix1, matrix2);
  Matrix result = Matrix(shape, matrix1.device(), matrix1.data_type());
  div_dispatcher(matrix1, matrix2, result);
  return result;
}

Matrix add(const Scalar& scalar, const Matrix& matrix) {
  return add(scalar.to_matrix(matrix.device(), matrix.data_type()), matrix);
}
Matrix sub(const Scalar& scalar, const Matrix& matrix) {
  return sub(scalar.to_matrix(matrix.device(), matrix.data_type()), matrix);
}
Matrix mul(const Scalar& scalar, const Matrix& matrix) {
  return mul(scalar.to_matrix(matrix.device(), matrix.data_type()), matrix);
}
Matrix div(const Scalar& scalar, const Matrix& matrix) {
  return div(scalar.to_matrix(matrix.device(), matrix.data_type()), matrix);
}
Matrix add(const Matrix& matrix, const Scalar& scalar) {
  return add(scalar.to_matrix(matrix.device(), matrix.data_type()), matrix);
}
Matrix sub(const Matrix& matrix, const Scalar& scalar) {
  return sub(scalar.to_matrix(matrix.device(), matrix.data_type()), matrix);
}
Matrix mul(const Matrix& matrix, const Scalar& scalar) {
  return mul(scalar.to_matrix(matrix.device(), matrix.data_type()), matrix);
}
Matrix div(const Matrix& matrix, const Scalar& scalar) {
  return div(scalar.to_matrix(matrix.device(), matrix.data_type()), matrix);
}
}  // namespace starml