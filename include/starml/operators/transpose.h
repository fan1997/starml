#pragma once
#include "starml/basic/matrix.h"
#include "starml/basic/dispatch.h"
namespace starml {
using transpose_op_kernel_fn = void (*)(const Matrix& matrix1, Matrix& result);
STARML_DECLARE_DISPATCHER(transpose_dispatcher, transpose_op_kernel_fn);

Matrix transpose(const Matrix& matrix1);
Matrix& transpose(const Matrix& matrix1, Matrix& result);
}
