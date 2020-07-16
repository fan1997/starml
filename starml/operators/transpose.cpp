#include "starml/operators/transpose.h"

namespace starml {
  STARML_DEFINE_DISPATCHER(transpose_dispatcher);

  Matrix transpose(const Matrix& matrix1) {
    Matrix result =
        Matrix(matrix1.cols_num(), matrix1.rows_num(),
               matrix1.device_type().type(), matrix1.data_type().type());
    transpose_dispatcher(matrix1, result);
    return result;
  }
}
