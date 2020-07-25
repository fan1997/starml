#include "starml/preprocessing/scaler/standardscaler_op.h"
#include "starml/basic/common_cuda.h"
#include <iostream>


namespace starml {
namespace preprocessing {
namespace scaler {
// FIT
namespace {
template <typename T>
__global__ void fit_kernel(const T* data_ptr, T* mean_ptr, T* std_ptr, int rows_num, int cols_num) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < cols_num) {
      mean_ptr[i] = 0;
      for (int j = 0; j < rows_num; j++) {
          mean_ptr[i] +=  data_ptr[j * cols_num + i];
      }
      mean_ptr[i] /= rows_num;
      std_ptr[i] = 0;
      auto mean_i = mean_ptr[i];
      for (int j = 0; j < rows_num; j++) {
          auto diff =  data_ptr[j * cols_num + i] - mean_i;
          std_ptr[i] += diff * diff;
      }
      std_ptr[i] /= rows_num;
      std_ptr[i] = sqrt(std_ptr[i]);
  }
}

void fit_impl(const Matrix& origin_data, Matrix& mean_data,
              Matrix& std_data) {
  // std::cout << "cuda fit..." << '\n';
  auto data_type = origin_data.data_type().type();
  auto rows_num =  origin_data.dim(0);
  auto cols_num =  origin_data.dim(1);

  STARML_DISPATCH_FLOATING_TYPES(data_type, "FIT", [&]() {
    auto data_ptr = origin_data.data<scalar_t>();
    auto mean_ptr = mean_data.mutable_data<scalar_t>();
    auto std_ptr = std_data.mutable_data<scalar_t>();
    dim3 dimGrid(ceil(cols_num / 256.0), 1, 1);
    dim3 dimBlock(256, 1, 1);
    fit_kernel<scalar_t><<<dimGrid, dimBlock>>>(data_ptr, mean_ptr, std_ptr, rows_num, cols_num);
    cudaDeviceSynchronize();
  });
}

}  // namespace
STARML_REGISTER_KERNEL(stsfit_dispatcher, kCUDA, &fit_impl);

// trans
namespace {
template <typename T>
__global__ void trans_kernel(const T* data_ptr, const T* mean_ptr, const T* std_ptr, T* res_ptr, int rows_num, int cols_num) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < rows_num) {
      for (int j = 0; j < cols_num; j++) {
          res_ptr[i * cols_num + j] =  (data_ptr[i * cols_num + j] - mean_ptr[j]) / std_ptr[j];
      }
  }
}

void trans_impl(const Matrix& origin_data, Matrix& result,
                const Matrix& mean_data, const Matrix& std_data) {
  // std::cout << "cuda trans..." << '\n';
  auto data_type = origin_data.data_type().type();
  auto rows_num =  origin_data.dim(0);
  auto cols_num =  origin_data.dim(1);

  STARML_DISPATCH_TYPES(data_type, "TRANS", [&]() {
    auto data_ptr = origin_data.data<scalar_t>();
    auto mean_ptr = mean_data.data<scalar_t>();
    auto std_ptr = std_data.data<scalar_t>();
    auto res_ptr = result.mutable_data<scalar_t>();
    dim3 dimGrid(ceil(rows_num / 256.0), 1, 1);
    dim3 dimBlock(256, 1, 1);
    trans_kernel<scalar_t><<<dimGrid, dimBlock>>>(data_ptr,  mean_ptr, std_ptr,  res_ptr, rows_num, cols_num);
    cudaDeviceSynchronize();
  });
}

}  // namespace
STARML_REGISTER_KERNEL(ststrans_dispatcher, kCUDA, &trans_impl);

namespace {
template <typename T>
__global__ void invtrans_kernel(const T* data_ptr, const T* mean_ptr, const T* std_ptr, T* res_ptr, int rows_num, int cols_num) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < rows_num) {
      for (int j = 0; j < cols_num; j++) {
          res_ptr[i * cols_num + j] =  data_ptr[i * cols_num + j] * std_ptr[j] + mean_ptr[j];
      }
  }
}

void invtrans_impl(const Matrix& transformed_data, Matrix& result,
                const Matrix& mean_data, const Matrix& std_data) {
  // std::cout << "cuda inv trans..." << '\n';
  auto data_type = transformed_data.data_type().type();
  auto rows_num =  transformed_data.dim(0);
  auto cols_num =  transformed_data.dim(1);

  STARML_DISPATCH_TYPES(data_type, "INVTRANS", [&]() {
    auto data_ptr = transformed_data.data<scalar_t>();
    auto mean_ptr = mean_data.data<scalar_t>();
    auto std_ptr = std_data.data<scalar_t>();
    auto res_ptr = result.mutable_data<scalar_t>();
    dim3 dimGrid(ceil(rows_num / 256.0), 1, 1);
    dim3 dimBlock(256, 1, 1);
    invtrans_kernel<scalar_t><<<dimGrid, dimBlock>>>(data_ptr,  mean_ptr, std_ptr,  res_ptr, rows_num, cols_num);
    cudaDeviceSynchronize();
  });
}

}  // namespace
STARML_REGISTER_KERNEL(stsinvtrans_dispatcher, kCUDA, &invtrans_impl);


}
}
}  // namespace starml
