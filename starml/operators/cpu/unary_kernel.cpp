#include <cmath>
#include <omp.h>
#include "starml/operators/unary_ops.h"

namespace starml {
namespace {

template <typename TScalarType, typename TResultType, typename TOp>
void eval_unary(const TScalarType* data, TResultType* result_data, int start,
                int end, TOp op) {
#pragma omp parallel for
  for (int i = start; i < end; i++) {
    *(result_data + i) = op(*(data + i));
  }
}

void exp_impl(const Matrix& matrix, Matrix& result) {
  auto dtype = matrix.data_type().type();
  auto result_dtype = result.data_type().type();
  STARML_DISPATCH_TYPES(dtype, "EXP_CPU", [&]() {
    auto data = matrix.data<scalar_t>();
    using scalar_type = scalar_t;
    STARML_DISPATCH_TYPES(result_dtype, "EXP_CPU", [&]() {
      auto result_data = result.mutable_data<scalar_t>();
      using result_scalar_type = scalar_t;
      eval_unary(
          data, result_data, 0, result.size(),
          [=](scalar_type a) -> result_scalar_type { return std::exp(a); });
    });
  });
}

void log_impl(const Matrix& matrix, Matrix& result) {
  auto dtype = matrix.data_type().type();
  auto result_dtype = result.data_type().type();
  STARML_DISPATCH_TYPES(dtype, "LOG_CPU", [&]() {
    auto data = matrix.data<scalar_t>();
    using scalar_type = scalar_t;
    STARML_DISPATCH_TYPES(result_dtype, "LOG_CPU", [&]() {
      auto result_data = result.mutable_data<scalar_t>();
      using result_scalar_type = scalar_t;
      eval_unary(
          data, result_data, 0, result.size(),
          [=](scalar_type a) -> result_scalar_type { return std::log(a); });
    });
  });
}

void negtive_impl(const Matrix& matrix, Matrix& result) {
  auto dtype = matrix.data_type().type();
  auto result_dtype = result.data_type().type();
  STARML_DISPATCH_TYPES(dtype, "NEG_CPU", [&]() {
    auto data = matrix.data<scalar_t>();
    using scalar_type = scalar_t;
    STARML_DISPATCH_TYPES(result_dtype, "NEG_CPU", [&]() {
      auto result_data = result.mutable_data<scalar_t>();
      using result_scalar_type = scalar_t;
      eval_unary(
          data, result_data, 0, result.size(),
          [=](scalar_type a) -> result_scalar_type { return -a; });
    });
  });
}

}  // namespace

STARML_REGISTER_KERNEL(exp_dispatcher, &exp_impl, kCPU, kCPU);
STARML_REGISTER_KERNEL(log_dispatcher, &log_impl, kCPU, kCPU);
STARML_REGISTER_KERNEL(negtive_dispatcher, &negtive_impl, kCPU, kCPU);

}  // namespace starml
