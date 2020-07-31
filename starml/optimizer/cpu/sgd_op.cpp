#include "starml/optimizer/SGD.h"

namespace starml {
namespace optimizer{

namespace {
template <typename T>
void sgd_op_impl_kernel(T* param, const T* grad, const float lr, int size){
#pragma omp parallel for
  for (int i = 0; i < size; i++) {
      param[i] -= lr * grad[i];
  }
}
void sgd_op_impl(Matrix& parameters,  Matrix& grad, const float lr) {
  auto data_type = parameters.data_type().type();
  int size = parameters.size();
  STARML_DISPATCH_TYPES(data_type, "SGD", [&]() {
    auto param_ptr = parameters.mutable_data<scalar_t>();
    auto grad_ptr = grad.data<scalar_t>();
    sgd_op_impl_kernel(param_ptr, grad_ptr, lr, size);
  });
}

}  // namespace

STARML_REGISTER_KERNEL(sgd_dispatcher, &sgd_op_impl, kCPU, kCPU);
}  // namespace optimizer
}  // namespace starml
