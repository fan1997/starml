#include "starml/operators/unary_ops.h"

namespace starml {

  STARML_DEFINE_DISPATCHER(exp_dispatcher);
  Matrix exp(const Matrix& matrix1) {
    Matrix result = Matrix(matrix1.dims(), matrix1.device(), matrix1.data_type());
    // 
    exp_dispatcher(matrix1, result);
    return result;
  }

  STARML_DEFINE_DISPATCHER(sqrt_dispatcher);
  Matrix sqrt(const Matrix& matrix1) {
    Matrix result = Matrix(matrix1.dims(), matrix1.device(), matrix1.data_type());
    sqrt_dispatcher(matrix1, result);
    return result;
  }

  STARML_DEFINE_DISPATCHER(square_dispatcher);
  Matrix square(const Matrix& matrix1) {
    Matrix result = Matrix(matrix1.dims(), matrix1.device(), matrix1.data_type());
    square_dispatcher(matrix1, result);
    return result;
  }

}
