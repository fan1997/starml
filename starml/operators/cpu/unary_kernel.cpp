#include <cmath>
#include "starml/operators/unary_ops.h"

namespace starml {
namespace {

template <typename TScalarType, typename TOp>
void eval_unary(const TScalarType* data, TScalarType* result_data, int start,
                int end, TOp op) {
  for (int i = start; i < end; i++) {
    *(result_data + i) = op(*(data + i));
  }
}

void exp_impl(const Matrix& matrix, Matrix& result) {
  auto data_type = matrix.data_type().type();
  STARML_DISPATCH_TYPES(data_type, "CPU_EXP", [&]() {
    auto data = matrix.data<scalar_t>();
    auto result_data = result.mutable_data<scalar_t>();
    eval_unary<scalar_t>(data, result_data, 0, result.size(),
                         [=](scalar_t a) -> scalar_t { return std::exp(a); });
  });
}

}  // namespace

STARML_REGISTER_KERNEL(exp_dispatcher, kCPU, &exp_impl);

}  // namespace starml