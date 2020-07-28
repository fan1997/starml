#include "starml/preprocessing/scaler/standardscaler_op.h"
#include <omp.h>
#include <cmath>

namespace starml {
namespace preprocessing {
namespace scaler {

namespace {
//fit
template <typename T>
void fit_impl_kernel(const T* data_ptr, T* mean_ptr, T* std_ptr, int rows_num, int cols_num){
#pragma omp parallel for
  for (int i = 0; i < cols_num; i++) {
      mean_ptr[i] = 0;
      for (int j = 0; j < rows_num; j++) {
          mean_ptr[i] +=  data_ptr[j * cols_num + i];
      }
      mean_ptr[i] /= rows_num;
  }
#pragma omp parallel for
  for (int i = 0; i < cols_num; i++) {
      std_ptr[i] = 0;
      auto mean_i = mean_ptr[i];
      for (int j = 0; j < rows_num; j++) {
          auto diff =  data_ptr[j * cols_num + i] - mean_i;
          std_ptr[i] += diff * diff;
      }
      std_ptr[i] /= rows_num;
      std_ptr[i] = std::sqrt(std_ptr[i]);
  }
}
void fit_impl(const Matrix& origin_data, Matrix& mean_data,
              Matrix& std_data) {
  // std::cout << "cpu fit..." << '\n';   //check rows_num
  auto data_type = origin_data.data_type().type();
  auto rows_num =  origin_data.dim(0);
  auto cols_num =  origin_data.dim(1);

  STARML_DISPATCH_TYPES(data_type, "FIT", [&]() {
    auto data_ptr = origin_data.data<scalar_t>();
    auto mean_ptr = mean_data.mutable_data<scalar_t>();
    auto std_ptr = std_data.mutable_data<scalar_t>();
    fit_impl_kernel(data_ptr, mean_ptr, std_ptr, rows_num, cols_num);
  });
}

//trans
template <typename T>
void trans_impl_kernel(const T* data_ptr, const T* mean_ptr, const T* std_ptr, T* res_ptr, int rows_num, int cols_num){
#pragma omp parallel for
    for (int i = 0; i < rows_num; i++) {
        for (int j = 0; j < cols_num; j++) {
            res_ptr[i * cols_num + j] =  (data_ptr[i * cols_num + j] - mean_ptr[j]) / std_ptr[j];
        }
    }
}
void trans_impl(const Matrix& origin_data, Matrix& result,
                const Matrix& mean_data, const Matrix& std_data) {
  // std::cout << "cpu trans..." << '\n';   //check rows_num
  auto data_type = origin_data.data_type().type();
  auto rows_num =  origin_data.dim(0);
  auto cols_num =  origin_data.dim(1);
  STARML_DISPATCH_TYPES(data_type, "TRANS", [&]() {
    auto data_ptr = origin_data.data<scalar_t>();
    auto mean_ptr = mean_data.data<scalar_t>();
    auto std_ptr = std_data.data<scalar_t>();
    auto res_ptr = result.mutable_data<scalar_t>();
    trans_impl_kernel(data_ptr,  mean_ptr, std_ptr,  res_ptr, rows_num, cols_num);
  });
}

// inv trans
template <typename T>
void invtrans_impl_kernel(const T* data_ptr, const T* mean_ptr, const T* std_ptr, T* res_ptr, int rows_num, int cols_num){
#pragma omp parallel for
    for (int i = 0; i < rows_num; i++) {
        for (int j = 0; j < cols_num; j++) {
            res_ptr[i * cols_num + j] =  data_ptr[i * cols_num + j] * std_ptr[j] + mean_ptr[j];
        }
    }
}
void invtrans_impl(const Matrix& transformed_data, Matrix& result,
                const Matrix& mean_data, const Matrix& std_data) {
  // std::cout << "cpu inv trans..." << '\n';   //check rows_num
  auto data_type = transformed_data.data_type().type();
  auto rows_num =  transformed_data.dim(0);
  auto cols_num =  transformed_data.dim(1);
  STARML_DISPATCH_TYPES(data_type, "INVTRANS", [&]() {
    auto data_ptr = transformed_data.data<scalar_t>();
    auto mean_ptr = mean_data.data<scalar_t>();
    auto std_ptr = std_data.data<scalar_t>();
    auto res_ptr = result.mutable_data<scalar_t>();
    invtrans_impl_kernel(data_ptr, mean_ptr, std_ptr, res_ptr, rows_num, cols_num);
  });
}
}  // namespace

STARML_REGISTER_KERNEL(stsfit_dispatcher, &fit_impl, kCPU, kCPU);
STARML_REGISTER_KERNEL(ststrans_dispatcher, &trans_impl, kCPU, kCPU);
STARML_REGISTER_KERNEL(stsinvtrans_dispatcher, &invtrans_impl, kCPU, kCPU);

}
}
}
