#include "starml/operators/concat.h"

namespace starml {

  STARML_DEFINE_DISPATCHER(concat_dispatcher);
  Matrix concat(const Matrix& matrix1, const Matrix& matrix2, int axis) { //default Vertical
    axis = 1;
    auto m1_rows_num =  matrix1.rows_num();
    auto m1_cols_num =  matrix1.cols_num();
    auto m2_rows_num =  matrix2.rows_num();
    auto m2_cols_num =  matrix2.cols_num();
    //check
    //check(axis == 0 || axis == 1)
    // check(axis == 0 ?  m1_cols_num == m2_cols_num : m1_rows_num == m2_rows_num)
    auto res_rows_num = axis == 0 ? m1_rows_num + m2_rows_num : m1_rows_num;
    auto res_cols_num = axis == 0 ? m1_cols_num : m1_cols_num + m2_cols_num;
    Matrix result =
        Matrix(res_rows_num, res_cols_num,
               matrix1.device_type().type(), matrix1.data_type().type());
    // concat_dispatcher(matrix1, matrix2, result, axis);
    concat_dispatcher(matrix1, matrix2, result);
    return result;
  }

}
