#include "starml/modelevaluator/metrics/mse_op.h"
#include "starml/basic/common_cuda.h"
#include <thrust/transform_reduce.h>
#include <thrust/system/cuda/detail/par.h>

namespace starml {
namespace modelevaluator {
namespace metrics{

namespace {
template <typename T>
__global__ void mse_kernel(const T* y, const T* y_pred, int nums, float* m_diff_ptr) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < nums) {
    float diff = y[i] - y_pred[i];
    m_diff_ptr[i] = diff * diff;
  }
}

float mse_impl(const Matrix& y, const Matrix& y_pred) {
  auto data_type = y.data_type().type();
  int nums = y.rows_num() * y.cols_num();
  Matrix m_diff(nums, 1, kCUDA, kFloat);
  float* m_diff_ptr = m_diff.data<float>();
  float score = 0.0;
  STARML_DISPATCH_TYPES(data_type, "MSE", [&]() {
    scalar_t *y_ptr = y.data<scalar_t>();
    scalar_t *y_pred_ptr = y_pred.data<scalar_t>();
    dim3 dimGrid(ceil(nums / 256.0), 1, 1);
    dim3 dimBlock(256, 1, 1);
    mse_kernel<scalar_t><<<dimGrid, dimBlock>>>(y_ptr, y_pred_ptr, nums, m_diff_ptr);
    score = thrust::reduce(thrust::cuda::par, m_diff_ptr, m_diff_ptr + nums, (float) 0, thrust::plus<float>());
  });
  return score / nums;
}

}  // namespace
STARML_REGISTER_KERNEL(mse_dispatcher, kCUDA, &mse_impl);

}
}
}  // namespace starml
