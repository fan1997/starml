#pragma once
#include "starml/basic/matrix.h"
#include "starml/basic/dispatch.h"
namespace starml {
using concat_kernel_fn = void (*)(const Matrix& matrix1,
                                     const Matrix& matrix2, Matrix& result, int axis);
// using concat_kernel_fn = void (*)(const Matrix& matrix1,
//                                      const Matrix& matrix2, Matrix& result);
STARML_DECLARE_DISPATCHER(concat_dispatcher, concat_kernel_fn);
Matrix concat(const Matrix& matrix1, const Matrix& matrix2, int axis = 0);
// Matrix concat(const Matrix& matrix1, const Matrix& matrix2);
}
