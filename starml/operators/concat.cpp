#include "starml/operators/concat.h"

namespace starml {

  STARML_DEFINE_DISPATCHER(concat_dispatcher);
  Matrix concat(const Matrix& matrix1, const Matrix& matrix2, int axis) { //default Vertical
    auto m1_rows_num =  matrix1.dim(0);
    auto m1_cols_num =  matrix1.dim(1);
    auto m2_rows_num =  matrix2.dim(0);
    auto m2_cols_num =  matrix2.dim(1);
    //check
    //check(axis == 0 || axis == 1)
    // check(axis == 0 ?  m1_cols_num == m2_cols_num : m1_rows_num == m2_rows_num)
    auto res_rows_num = axis == 0 ? m1_rows_num + m2_rows_num : m1_rows_num;
    auto res_cols_num = axis == 0 ? m1_cols_num : m1_cols_num + m2_cols_num;
    Matrix result =
        Matrix({res_rows_num, res_cols_num},
               matrix1.device(), matrix1.data_type());
    concat_dispatcher(matrix1, matrix2, result, axis);
    // concat_dispatcher(matrix1, matrix2, result);
    return result;
  }

}
