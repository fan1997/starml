#include "starml/optimizer/SGD.h"
#include "starml/basic/common_cuda.h"
namespace starml {
namespace optimizer{

namespace {
template <typename T>
__global__ void sgd_op_impl_kernel(T* param, const T* grad, const float lr, int size){
  STARML_CUDA_1D_KERNEL_LOOP(i, size){
      param[i] -= lr * grad[i];
  }
}
void sgd_op_impl(Matrix& parameters,  const Matrix& grad, const float lr) {
  auto data_type = parameters.data_type().type();
  int size = parameters.size();
  STARML_DISPATCH_TYPES(data_type, "SGD", [&]() {
    auto param_ptr = parameters.mutable_data<scalar_t>();
    auto grad_ptr = grad.data<scalar_t>();
    dim3 dimGrid(ceil(size / 256.0), 1, 1);
    dim3 dimBlock(256, 1, 1);
    sgd_op_impl_kernel<scalar_t><<<dimGrid, dimBlock>>>(param_ptr, grad_ptr, lr, size);
    cudaDeviceSynchronize();
  });
}

}  // namespace

STARML_REGISTER_KERNEL(sgd_dispatcher, &sgd_op_impl, kCUDA, kCUDA);
}  // namespace optimizer
}  // namespace starml
