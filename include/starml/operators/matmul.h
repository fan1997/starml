#pragma once
#include "starml/basic/matrix.h"
#include "starml/basic/dispatch.h"
namespace starml {

enum class MatmulOption: int16_t {
  NoTrans = 100,
  Trans = 101,
  ConjTrans = 102
};
constexpr MatmulOption kNoTrans = MatmulOption::NoTrans;
constexpr MatmulOption kTrans = MatmulOption::Trans;
constexpr MatmulOption kConjTrans = MatmulOption::ConjTrans;

using matmul_op_kernel_fn = void (*)(const Matrix& matrix1,
                                     const Matrix& matrix2, Matrix& result);
STARML_DECLARE_DISPATCHER(matmul_dispatcher, matmul_op_kernel_fn);

Matrix matmul(const Matrix& matrix1, const Matrix& matrix2, MatmulOption option_1 = kNoTrans, MatmulOption option_2 = kNoTrans);

}
