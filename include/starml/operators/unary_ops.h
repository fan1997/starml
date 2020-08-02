#pragma once
#include "starml/basic/matrix.h"
#include "starml/basic/dispatch.h"
namespace starml {
using unary_op_kernel_fn = void (*)(const Matrix& matrix, Matrix& result, Handle* handle);
STARML_DECLARE_DISPATCHER(exp_dispatcher, unary_op_kernel_fn);
STARML_DECLARE_DISPATCHER(cast_dispatcher, unary_op_kernel_fn);
STARML_DECLARE_DISPATCHER(log_dispatcher, unary_op_kernel_fn);
STARML_DECLARE_DISPATCHER(negtive_dispatcher, unary_op_kernel_fn);

Matrix exp(const Matrix& matrix, Handle* handle = NULL);
Matrix exp(const Matrix& matrix, Matrix& result, Handle* handle = NULL);

Matrix log(const Matrix& matrix, Handle* handle = NULL);
Matrix log(const Matrix& matrix, Matrix& result, Handle* handle = NULL);

Matrix negtive(const Matrix& matrix, Handle* handle = NULL);
Matrix negtive(const Matrix& matrix, Matrix& result, Handle* handle = NULL);
Matrix cast(const Matrix& matrix, const DataType& data_type, Handle* handle = NULL);
Matrix cast(const Matrix& matrix, Matrix& result, Handle* handle = NULL);
}
