#include "starml/preprocessing/scaler/standardscaler_op.h"
#include "starml/basic/common_cuda.h"
#include <iostream>
// #include <thrust/transform_reduce.h>
// #include <thrust/system/cuda/detail/par.h>

namespace starml {
namespace preprocessing {
namespace scaler {
// FIT
namespace {
template <typename T>
__global__ void fit_kernel(const T* y, const T* y_pred, int nums, float* m_diff_ptr) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < nums) {
    float diff = y[i] - y_pred[i];
    m_diff_ptr[i] = diff * diff;
  }
}

void fit_impl(const Matrix& origin_data, Matrix& mean_data,
              Matrix& std_data) {
  std::cout << "cuda fit..." << '\n';
  auto data_type = origin_data.data_type().type();
  int nums = origin_data.rows_num() * origin_data.cols_num();
}

}  // namespace
STARML_REGISTER_KERNEL(stsfit_dispatcher, kCUDA, &fit_impl);

// trans
namespace {
template <typename T>
__global__ void trans_kernel(const T* y, const T* y_pred, int nums, float* m_diff_ptr) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < nums) {
    float diff = y[i] - y_pred[i];
    m_diff_ptr[i] = diff * diff;
  }
}

void trans_impl(const Matrix& origin_data, Matrix& result,
                const Matrix& mean_data, const Matrix& std_data) {
  std::cout << "cuda trans..." << '\n';
  auto data_type = origin_data.data_type().type();
  int nums = origin_data.rows_num() * origin_data.cols_num();
}

}  // namespace
STARML_REGISTER_KERNEL(ststrans_dispatcher, kCUDA, &trans_impl);

namespace {
template <typename T>
__global__ void invtrans_kernel(const T* y, const T* y_pred, int nums, float* m_diff_ptr) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < nums) {
    float diff = y[i] - y_pred[i];
    m_diff_ptr[i] = diff * diff;
  }
}

void invtrans_impl(const Matrix& transformed_data, Matrix& result,
                const Matrix& mean_data, const Matrix& std_data) {
  std::cout << "cuda inv trans..." << '\n';
  auto data_type = transformed_data.data_type().type();
  int nums = transformed_data.rows_num() * transformed_data.cols_num();
}

}  // namespace
STARML_REGISTER_KERNEL(stsinvtrans_dispatcher, kCUDA, &invtrans_impl);


}
}
}  // namespace starml
