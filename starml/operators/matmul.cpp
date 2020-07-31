#include "starml/operators/transpose.h"
#include "starml/operators/matmul.h"

namespace starml {
  STARML_DEFINE_DISPATCHER(matmul_dispatcher);

  Matrix matmul(const Matrix& matrix1, const Matrix& matrix2, MatmulOption option_1, MatmulOption option_2) {

    auto m1_rows_num =  option_1 == kNoTrans ? matrix1.dim(0) : matrix1.dim(1);
    auto m1_cols_num =  option_1 == kNoTrans ? matrix1.dim(1) : matrix1.dim(0);
    auto m2_rows_num =  option_2 == kNoTrans ? matrix2.dim(0) : matrix2.dim(1);
    auto m2_cols_num =  option_2 == kNoTrans ? matrix2.dim(1) : matrix2.dim(0);
    STARML_CHECK_EQ(m1_cols_num, m2_rows_num) << "Dims not match in matmul";

    Matrix matrix1_new = option_1 == kNoTrans ? matrix1 : transpose(matrix1);
    Matrix matrix2_new = option_2 == kNoTrans ? matrix2 : transpose(matrix2);

    Matrix result =
        Matrix({matrix1_new.dim(0), matrix2_new.dim(1)},
               matrix1_new.device(), matrix1_new.data_type());
    matmul_dispatcher(matrix1_new, matrix2_new, result);
    return result;
  }

  Matrix& matmul(const Matrix& matrix1, const Matrix& matrix2, Matrix& result,
                MatmulOption option_1, MatmulOption option_2) {
    auto m1_rows_num =  option_1 == kNoTrans ? matrix1.dim(0) : matrix1.dim(1);
    auto m1_cols_num =  option_1 == kNoTrans ? matrix1.dim(1) : matrix1.dim(0);
    auto m2_rows_num =  option_2 == kNoTrans ? matrix2.dim(0) : matrix2.dim(1);
    auto m2_cols_num =  option_2 == kNoTrans ? matrix2.dim(1) : matrix2.dim(0);
    STARML_CHECK_EQ(m1_cols_num, m2_rows_num) << "Dims not match in matmul";
    STARML_CHECK((result.dim(0) == m1_rows_num) && (result.dim(1) == m2_cols_num))
        << "Dimension of result for inplace operator should be well "
           "preallocated.";

    Matrix matrix1_new = option_1 == kNoTrans ? matrix1 : transpose(matrix1);
    Matrix matrix2_new = option_2 == kNoTrans ? matrix2 : transpose(matrix2);

    matmul_dispatcher(matrix1_new, matrix2_new, result);

    return result;
  }
}
