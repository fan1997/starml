#pragma once
#include "starml/basic/matrix.h"
#include "starml/basic/dispatch.h"
namespace starml {
using unary_op_kernel_fn = void (*)(const Matrix& matrix1, Matrix& result);
STARML_DECLARE_DISPATCHER(exp_dispatcher, unary_op_kernel_fn);
STARML_DECLARE_DISPATCHER(sqrt_dispatcher, unary_op_kernel_fn);
STARML_DECLARE_DISPATCHER(square_dispatcher, unary_op_kernel_fn);
Matrix exp(const Matrix& matrix1);
Matrix sqrt(const Matrix& matrix1);
Matrix square(const Matrix& matrix1);
}
