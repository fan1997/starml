#pragma once
#include "starml/basic/matrix.h"
#include "starml/basic/dispatch.h"
namespace starml {
using unary_op_kernel_fn = void (*)(const Matrix& matrix, Matrix& result);
STARML_DECLARE_DISPATCHER(exp_dispatcher, unary_op_kernel_fn);
STARML_DECLARE_DISPATCHER(log_dispatcher, unary_op_kernel_fn);
STARML_DECLARE_DISPATCHER(negtive_dispatcher, unary_op_kernel_fn);

Matrix exp(const Matrix& matrix);
Matrix exp(const Matrix& matrix, Matrix& result);

Matrix log(const Matrix& matrix);
Matrix log(const Matrix& matrix, Matrix& result);

Matrix negtive(const Matrix& matrix);
Matrix negtive(const Matrix& matrix, Matrix& result);
}
