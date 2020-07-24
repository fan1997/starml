#pragma once
#include "starml/basic/dispatch.h"
#include "starml/basic/matrix.h"
#include "starml/basic/scalar.h"
namespace starml {
using binary_op_kernel_fn = void (*)(const Matrix& matrix1,
                                     const Matrix& matrix2, Matrix& result,
                                     bool blocking);
STARML_DECLARE_DISPATCHER(add_dispatcher, binary_op_kernel_fn);
STARML_DECLARE_DISPATCHER(sub_dispatcher, binary_op_kernel_fn);
STARML_DECLARE_DISPATCHER(mul_dispatcher, binary_op_kernel_fn);
STARML_DECLARE_DISPATCHER(div_dispatcher, binary_op_kernel_fn);

Matrix add(const Matrix& matrix1, const Matrix& matrix2, bool blocking = true);
Matrix sub(const Matrix& matrix1, const Matrix& matrix2, bool blocking = true);
Matrix mul(const Matrix& matrix1, const Matrix& matrix2, bool blocking = true);
Matrix div(const Matrix& matrix1, const Matrix& matrix2, bool blocking = true);

Matrix add(const Matrix& matrix1, const Matrix& matrix2, Matrix& result,
           bool blocking = true);
Matrix sub(const Matrix& matrix1, const Matrix& matrix2, Matrix& result,
           bool blocking = true);
Matrix mul(const Matrix& matrix1, const Matrix& matrix2, Matrix& result,
           bool blocking = true);
Matrix div(const Matrix& matrix1, const Matrix& matrix2, Matrix& result,
           bool blocking = true);

Matrix add(const Scalar& scalar, const Matrix& matrix, bool blocking = true);
Matrix sub(const Scalar& scalar, const Matrix& matrix, bool blocking = true);
Matrix mul(const Scalar& scalar, const Matrix& matrix, bool blocking = true);
Matrix div(const Scalar& scalar, const Matrix& matrix, bool blocking = true);

Matrix add(const Scalar& scalar, const Matrix& matrix, Matrix& result,
           bool blocking = true);
Matrix sub(const Scalar& scalar, const Matrix& matrix, Matrix& result,
           bool blocking = true);
Matrix mul(const Scalar& scalar, const Matrix& matrix, Matrix& result,
           bool blocking = true);
Matrix div(const Scalar& scalar, const Matrix& matrix, Matrix& result,
           bool blocking = true);

Matrix add(const Matrix& matrix, const Scalar& scalar, bool blocking = true);
Matrix sub(const Matrix& matrix, const Scalar& scalar, bool blocking = true);
Matrix mul(const Matrix& matrix, const Scalar& scalar, bool blocking = true);
Matrix div(const Matrix& matrix, const Scalar& scalar, bool blocking = true);

Matrix add(const Matrix& matrix, const Scalar& scalar, Matrix& result,
           bool blocking = true);
Matrix sub(const Matrix& matrix, const Scalar& scalar, Matrix& result,
           bool blocking = true);
Matrix mul(const Matrix& matrix, const Scalar& scalar, Matrix& result,
           bool blocking = true);
Matrix div(const Matrix& matrix, const Scalar& scalar, Matrix& result,
           bool blocking = true);
}  // namespace starml