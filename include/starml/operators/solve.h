#pragma once
#include "starml/basic/matrix.h"
#include "starml/basic/dispatch.h"
namespace starml {
using lu_solve_op_kernel_fn = void (*)(const Matrix& matrix1, const Matrix& matrix2, Matrix& result);
STARML_DECLARE_DISPATCHER(lu_solve_dispatcher, lu_solve_op_kernel_fn);
Matrix lu_solve(const Matrix& matrix1, const Matrix& matrix2);
Matrix& lu_solve(const Matrix& matrix1, const Matrix& matrix2, Matrix& result);
}
