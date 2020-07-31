#include "starml/operators/solve.h"

namespace starml {
  STARML_DEFINE_DISPATCHER(lu_solve_dispatcher);

  Matrix lu_solve(const Matrix& matrix1, const Matrix& matrix2) {
    //check matrix1 square
    STARML_CHECK_EQ(matrix1.dim(0), matrix1.dim(1)) << "matrix A must be square";
    //check matrix2
    STARML_CHECK_EQ(matrix2.dim(1), 1) << "matrix b must be {n , 1} shape";
    STARML_CHECK_EQ(matrix1.dim(0), matrix2.dim(0)) << "matrix A and b must have the same rows_num";
    Matrix result =
        Matrix({matrix1.dim(0), 1},
               matrix1.device(), matrix1.data_type());
    lu_solve_dispatcher(matrix1, matrix2, result);
    return result;
  }
// out of place
  Matrix& lu_solve(const Matrix& matrix1, const Matrix& matrix2, Matrix& result) {
    //check matrix1 square
    STARML_CHECK_EQ(matrix1.dim(0), matrix1.dim(1)) << "matrix A must be square";
    //check matrix2
    STARML_CHECK_EQ(matrix2.dim(1), 1) << "matrix b must be {n , 1} shape";
    STARML_CHECK_EQ(matrix1.dim(0), matrix2.dim(0)) << "matrix A and b must have the same rows_num";
    STARML_CHECK((result.dim(0) == matrix1.dim(0)) && (result.dim(1) == 1))
        << "Dimension of result for inplace operator should be well "
           "preallocated.";
    lu_solve_dispatcher(matrix1, matrix2, result);
    return result;
  }
}
