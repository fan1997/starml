#include "starml/optimizer/momentum.h"
#include "starml/basic/common_cuda.h"
namespace starml {
namespace optimizer{

namespace {
template <typename T>
__global__ void momentum_op_impl_kernel(T* param, const T* grad, T* accumulation,  float lr,  float momentum, int size){
  STARML_CUDA_1D_KERNEL_LOOP(i, size){
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
    dim3 dimGrid(ceil(size / 256.0), 1, 1);
    dim3 dimBlock(256, 1, 1);
    momentum_op_impl_kernel<scalar_t><<<dimGrid, dimBlock>>>(param_ptr, grad_ptr, accumulation_ptr, lr, momentum, size);
    cudaDeviceSynchronize();
  });
}

}  // namespace

STARML_REGISTER_KERNEL(momentum_dispatcher, &momentum_op_impl, kCUDA, kCUDA, kCUDA);
}  // namespace optimizer
}  // namespace starml
