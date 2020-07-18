#include "starml/operators/binary_ops.h"

namespace starml {
  STARML_DEFINE_DISPATCHER(add_dispatcher);

  Matrix add(const Matrix& matrix1, const Matrix& matrix2) {
    Matrix result =
        Matrix(matrix1.dims(), matrix1.device(), matrix1.data_type());
    add_dispatcher(matrix1, matrix2, result);
    return result;
  }
  STARML_DEFINE_DISPATCHER(sub_dispatcher);

  Matrix sub(const Matrix& matrix1, const Matrix& matrix2) {
    Matrix result =
        Matrix(matrix1.dims(), matrix1.device(), matrix1.data_type());
    sub_dispatcher(matrix1, matrix2, result);
    return result;
  }
}
