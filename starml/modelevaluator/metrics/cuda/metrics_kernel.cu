#include "starml/modelevaluator/metrics/metrics_op.h"
#include "starml/basic/common_cuda.h"
#include <thrust/transform_reduce.h>
#include <thrust/system/cuda/detail/par.h>

namespace starml {
namespace modelevaluator {
namespace metrics{

// mse
namespace {
template <typename T>
__global__ void mse_kernel(const T* y, const T* y_pred, int sizes, float* m_diff_ptr) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < sizes) {
    float diff = y[i] - y_pred[i];
    m_diff_ptr[i] = diff * diff;
  }
}
float mse_impl(const Matrix& y, const Matrix& y_pred) {
  auto data_type = y.data_type().type();
  int sizes = y.size();
  Matrix m_diff({sizes, 1}, Device(kCUDA), DataType(kFloat));
  float* m_diff_ptr = m_diff.data<float>();
  float score = 0.0;
  STARML_DISPATCH_TYPES(data_type, "MSE", [&]() {
    scalar_t *y_ptr = y.data<scalar_t>();
    scalar_t *y_pred_ptr = y_pred.data<scalar_t>();
    dim3 dimGrid(ceil(sizes / 256.0), 1, 1);
    dim3 dimBlock(256, 1, 1);
    mse_kernel<scalar_t><<<dimGrid, dimBlock>>>(y_ptr, y_pred_ptr, sizes, m_diff_ptr);
    cudaDeviceSynchronize();
    score = thrust::reduce(thrust::cuda::par, m_diff_ptr, m_diff_ptr + sizes, (float) 0, thrust::plus<float>());
  });
  return score / sizes;
}
}  // namespace
STARML_REGISTER_KERNEL(mse_dispatcher, kCUDA, &mse_impl);

// acc
namespace {
template <typename T>
__global__ void acc_impl_kernel(T* data1_ptr, T* data2_ptr, int size, int* sum){
     int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size) {
      int sum_local = (data1_ptr[i] == data2_ptr[i]);
      atomicAdd(sum, sum_local);
    }
}
float acc_impl(const Matrix& y, const Matrix& y_pred) {
  auto data_type = y.data_type().type();
  int size = y.size();
  int sum = 0; // for atomic add
  Matrix sum_m({1,1}, kInt, kCUDA);
  STARML_DISPATCH_TYPES(data_type, "ACC", [&]() {
    auto data1_ptr = y.data<scalar_t>();
    auto data2_ptr = y_pred.data<scalar_t>();
    dim3 dimGrid(ceil(size / 256.0), 1, 1);
    dim3 dimBlock(256, 1, 1);

    acc_impl_kernel<scalar_t><<<dimGrid, dimBlock>>>(data1_ptr, data2_ptr, size, sum_m.data<int>());
    cudaDeviceSynchronize();
  });
  sum = sum_m.to(kCPU).data<int>()[0];
  return sum;
}
}  // namespace
STARML_REGISTER_KERNEL(acc_dispatcher, kCUDA, &acc_impl);

}
}
}  // namespace starml
