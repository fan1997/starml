#include "starml/operators/concat.h"

namespace starml {

  STARML_DEFINE_DISPATCHER(concat_dispatcher);
// in place
  Matrix concat(const Matrix& matrix1, const Matrix& matrix2, int axis) { //default Vertical
    auto m1_rows_num =  matrix1.dim(0);
    auto m1_cols_num =  matrix1.dim(1);
    auto m2_rows_num =  matrix2.dim(0);
    auto m2_cols_num =  matrix2.dim(1);
    //check
    STARML_CHECK((axis == 0 || axis == 1)) << "axis should be 0 (Vertical) or 1(Horizontal)";
    if(axis == 0)
       STARML_CHECK((m1_cols_num == m2_cols_num)) << "cols_num should be the same when concat with axis = 0";
    else
       STARML_CHECK((m1_rows_num == m2_rows_num)) << "rows_num should be the same when concat  with axis = 1";
    auto res_rows_num = axis == 0 ? m1_rows_num + m2_rows_num : m1_rows_num;
    auto res_cols_num = axis == 0 ? m1_cols_num : m1_cols_num + m2_cols_num;
    Matrix result =
        Matrix({res_rows_num, res_cols_num},
               matrix1.device(), matrix1.data_type());
    concat_dispatcher(matrix1, matrix2, result, axis);
    return result;
  }
// out of place
  Matrix& concat(const Matrix& matrix1, const Matrix& matrix2, Matrix& result, int axis) { //default Vertical
    auto m1_rows_num =  matrix1.dim(0);
    auto m1_cols_num =  matrix1.dim(1);
    auto m2_rows_num =  matrix2.dim(0);
    auto m2_cols_num =  matrix2.dim(1);
    //check
    STARML_CHECK((axis == 0 || axis == 1)) << "axis should be 0 (Vertical) or 1(Horizontal)";
    if(axis == 0)
       STARML_CHECK((m1_cols_num == m2_cols_num)) << "cols_num should be the same when concat with axis = 0";
    else
       STARML_CHECK((m1_rows_num == m2_rows_num)) << "rows_num should be the same when concat  with axis = 1";
    auto res_rows_num = axis == 0 ? m1_rows_num + m2_rows_num : m1_rows_num;
    auto res_cols_num = axis == 0 ? m1_cols_num : m1_cols_num + m2_cols_num;
    STARML_CHECK((result.dim(0) == res_rows_num) && (result.dim(1) == res_cols_num))
        << "Dimension of result for inplace operator should be well "
           "preallocated.";
    concat_dispatcher(matrix1, matrix2, result, axis);
    return result;
  }

}
