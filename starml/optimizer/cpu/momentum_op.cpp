#include "starml/optimizer/momentum.h"

namespace starml {
namespace optimizer{

namespace {
template <typename T>
void momentum_op_impl_kernel(T* param, const T* grad, T* accumulation, const float lr, const float momentum, int size){
#pragma omp parallel for
  for (int i = 0; i < size; i++) {
      accumulation[i] = momentum * accumulation[i] + grad[i];
      param[i] -= lr * accumulation[i];
  }
}
void momentum_op_impl(Matrix& parameters,  const Matrix& grad, Matrix& accumulation, const float lr, const float momentum) {
  auto data_type = parameters.data_type().type();
  int size = parameters.size();
  STARML_DISPATCH_TYPES(data_type, "Momentum", [&]() {
    auto param_ptr = parameters.mutable_data<scalar_t>();
    auto grad_ptr = grad.data<scalar_t>();
    auto accumulation_ptr = accumulation.mutable_data<scalar_t>();
    momentum_op_impl_kernel(param_ptr, grad_ptr, accumulation_ptr, lr, momentum, size);
  });
}

}  // namespace

STARML_REGISTER_KERNEL(momentum_dispatcher, &momentum_op_impl, kCPU, kCPU, kCPU);
}  // namespace optimizer
}  // namespace starml
