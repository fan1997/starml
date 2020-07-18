#include "starml/operators/transpose.h"

namespace starml {
  STARML_DEFINE_DISPATCHER(transpose_dispatcher);

  Matrix transpose(const Matrix& matrix1) {
    auto m1_rows_num =  matrix1.dim(0);
    auto m1_cols_num =  matrix1.dim(1);
    Matrix result =
        Matrix({m1_cols_num, m1_rows_num}, matrix1.device(), matrix1.data_type());
        transpose_dispatcher(matrix1, result);
    return result;
  }
}
