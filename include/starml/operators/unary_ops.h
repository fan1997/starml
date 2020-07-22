#pragma once
#include "starml/basic/matrix.h"
#include "starml/basic/dispatch.h"
namespace starml {
using unary_op_kernel_fn = void (*)(const Matrix& matrix, Matrix& result);
STARML_DECLARE_DISPATCHER(exp_dispatcher, unary_op_kernel_fn);

Matrix exp(const Matrix& matrix);

}