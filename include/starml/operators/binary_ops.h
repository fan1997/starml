#pragma once
#include "starml/basic/dispatch.h"
#include "starml/basic/matrix.h"
#include "starml/basic/scalar.h"
namespace starml {
using binary_op_kernel_fn = void (*)(const Matrix& matrix1,
                                     const Matrix& matrix2, Matrix& result);
STARML_DECLARE_DISPATCHER(add_dispatcher, binary_op_kernel_fn);
STARML_DECLARE_DISPATCHER(sub_dispatcher, binary_op_kernel_fn);
STARML_DECLARE_DISPATCHER(mul_dispatcher, binary_op_kernel_fn);
STARML_DECLARE_DISPATCHER(div_dispatcher, binary_op_kernel_fn);

Matrix add(const Matrix& matrix1, const Matrix& matrix2);
Matrix sub(const Matrix& matrix1, const Matrix& matrix2);
Matrix mul(const Matrix& matrix1, const Matrix& matrix2);
Matrix div(const Matrix& matrix1, const Matrix& matrix2);

Matrix add(const Matrix& matrix1, const Matrix& matrix2, Matrix& result);
Matrix sub(const Matrix& matrix1, const Matrix& matrix2, Matrix& result);
Matrix mul(const Matrix& matrix1, const Matrix& matrix2, Matrix& result);
Matrix div(const Matrix& matrix1, const Matrix& matrix2, Matrix& result);

Matrix add(const Scalar& scalar, const Matrix& matrix);
Matrix sub(const Scalar& scalar, const Matrix& matrix);
Matrix mul(const Scalar& scalar, const Matrix& matrix);
Matrix div(const Scalar& scalar, const Matrix& matrix);

Matrix add(const Scalar& scalar, const Matrix& matrix, Matrix& result);
Matrix sub(const Scalar& scalar, const Matrix& matrix, Matrix& result);
Matrix mul(const Scalar& scalar, const Matrix& matrix, Matrix& result);
Matrix div(const Scalar& scalar, const Matrix& matrix, Matrix& result);

Matrix add(const Matrix& matrix, const Scalar& scalar);
Matrix sub(const Matrix& matrix, const Scalar& scalar);
Matrix mul(const Matrix& matrix, const Scalar& scalar);
Matrix div(const Matrix& matrix, const Scalar& scalar);

Matrix add(const Matrix& matrix, const Scalar& scalar, Matrix& result);
Matrix sub(const Matrix& matrix, const Scalar& scalar, Matrix& result);
Matrix mul(const Matrix& matrix, const Scalar& scalar, Matrix& result);
Matrix div(const Matrix& matrix, const Scalar& scalar, Matrix& result);
}  // namespace starml