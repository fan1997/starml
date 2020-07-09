#pragma once
#include "starml/basic/matrix.h"
#include "starml/basic/dispatch.h"
namespace starml {
using binary_op_kernel_fn = void (*)(const Matrix& matrix1,
                                     const Matrix& matrix2, Matrix& result);
STARML_DECLARE_DISPATCHER(add_dispatcher, binary_op_kernel_fn);

Matrix add(const Matrix& matrix1, const Matrix& matrix2);

}