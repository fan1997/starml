#include "starml/operators/solve.h"

namespace starml {
  STARML_DEFINE_DISPATCHER(lu_solve_dispatcher);

  Matrix lu_solve(const Matrix& matrix1, const Matrix& matrix2) {
    //check matrix1 square
    Matrix result =
        Matrix({matrix1.dim(0), matrix1.dim(1)},
               matrix1.device(), matrix1.data_type());
    lu_solve_dispatcher(matrix1, matrix2, result);
    return result;
  }
}
