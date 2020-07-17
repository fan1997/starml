#include "starml/operators/solve.h"

namespace starml {
  STARML_DEFINE_DISPATCHER(lu_solve_dispatcher);

  Matrix lu_solve(const Matrix& matrix1, const Matrix& matrix2) {
    //check matrix1 square
    Matrix result =
        Matrix(matrix1.cols_num(), matrix1.rows_num(),
               matrix1.device_type().type(), matrix1.data_type().type());
    lu_solve_dispatcher(matrix1, matrix2, result);
    return result;
  }
}
