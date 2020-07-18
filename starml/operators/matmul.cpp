#include "starml/operators/transpose.h"
#include "starml/operators/matmul.h"

namespace starml {
  STARML_DEFINE_DISPATCHER(matmul_dispatcher);

  Matrix matmul(const Matrix& matrix1, const Matrix& matrix2, MatmulOption option_1, MatmulOption option_2) {
    auto m1_rows_num =  matrix1.dim(0);
    auto m1_cols_num =  matrix1.dim(1);
    auto m2_rows_num =  matrix2.dim(0);
    auto m2_cols_num =  matrix2.dim(1);
    Matrix matrix1_new = option_1 == kNoTrans ? matrix1 : transpose(matrix1);
    Matrix matrix2_new = option_2 == kNoTrans ? matrix2 : transpose(matrix2);
    Matrix result =
        Matrix({matrix1_new.dim(0), matrix2_new.dim(1)},
               matrix1_new.device(), matrix1_new.data_type());
    matmul_dispatcher(matrix1_new, matrix2_new, result);
    return result;
  }
}
