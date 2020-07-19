#pragma once
#include "starml/basic/matrix.h"
#include "starml/basic/dispatch.h"
namespace starml {

using add_scalar_op_kernel_fn = void (*)(const Matrix& matrix1, Matrix& result, double b);

STARML_DECLARE_DISPATCHER(add_scalar_dispatcher, add_scalar_op_kernel_fn);

template <typename T>
Matrix add_scalar(const Matrix& matrix1, T b) {
  auto data_type = matrix1.data_type().type();
  STARML_DISPATCH_TYPES(data_type, "CAST", [&]() {
    STARML_CHECK((sizeof(T) <= sizeof(double))) << "scalar should be lower than double";
    STARML_CHECK((sizeof(scalar_t) >= sizeof(T))) << "Temporary req: scalar must be lower than matrix type";
  });
  Matrix result =
      Matrix(matrix1.dims(), matrix1.device(), matrix1.data_type());
  double b_copy = b; // in order to not lose information
  add_scalar_dispatcher(matrix1, result, b_copy);
  return result;
}

}
