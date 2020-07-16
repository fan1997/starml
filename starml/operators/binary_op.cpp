#include "starml/operators/binary_ops.h"

namespace starml {
  STARML_DEFINE_DISPATCHER(add_dispatcher);

  Matrix add(const Matrix& matrix1, const Matrix& matrix2) {
    Matrix result =
        Matrix(matrix1.rows_num(), matrix1.cols_num(),
               matrix1.device_type().type(), matrix1.data_type().type());
    add_dispatcher(matrix1, matrix2, result);
    return result;
  }
  STARML_DEFINE_DISPATCHER(sub_dispatcher);

  Matrix sub(const Matrix& matrix1, const Matrix& matrix2) {
    Matrix result =
        Matrix(matrix1.rows_num(), matrix1.cols_num(),
               matrix1.device_type().type(), matrix1.data_type().type());
    sub_dispatcher(matrix1, matrix2, result);
    return result;
  }
}
