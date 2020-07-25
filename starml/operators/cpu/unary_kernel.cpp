#include "starml/operators/unary_ops.h"
#include <omp.h>
#include <cmath>

namespace starml {
namespace {

template <typename T>
void exp_impl_kernel(T* data_1, T* res_data, int& size){
#pragma omp parallel for
  for (int i = 0; i < size; i ++) {
    res_data[i] = std::exp(data_1[i]);
  }
}
void exp_impl(const Matrix& matrix1, Matrix& result) {
  auto data_type = matrix1.data_type().type();
  int size = matrix1.size();
  STARML_DISPATCH_FLOATING_TYPES(data_type, "EXP", [&]() {
    auto data1 = matrix1.data<scalar_t>();
    auto res_data = result.data<scalar_t>();
    exp_impl_kernel(data1, res_data, size);
  });
}

template <typename T>
void sqrt_impl_kernel(T* data_1, T* res_data, int& size){
#pragma omp parallel for
  for (int i = 0; i < size; i ++) {
    res_data[i] = std::sqrt(data_1[i]);
  }
}
void sqrt_impl(const Matrix& matrix1, Matrix& result) {
  auto data_type = matrix1.data_type().type();
  int size = matrix1.size();
  STARML_DISPATCH_FLOATING_TYPES(data_type, "SQRT", [&]() {
    auto data1 = matrix1.data<scalar_t>();
    auto res_data = result.data<scalar_t>();
    sqrt_impl_kernel(data1, res_data, size);
  });
}

template <typename T>
void square_impl_kernel(T* data_1, T* res_data, int& size){
#pragma omp parallel for
  for (int i = 0; i < size; i ++) {
    res_data[i] = data_1[i] * data_1[i];
  }
}
void square_impl(const Matrix& matrix1, Matrix& result) {
  auto data_type = matrix1.data_type().type();
  int size = matrix1.size();
  STARML_DISPATCH_TYPES(data_type, "SQUARE", [&]() {
    auto data1 = matrix1.data<scalar_t>();
    auto res_data = result.data<scalar_t>();
    square_impl_kernel(data1, res_data, size);
  });
}

}  // namespace

STARML_REGISTER_KERNEL(exp_dispatcher, kCPU, &exp_impl);
STARML_REGISTER_KERNEL(sqrt_dispatcher, kCPU, &sqrt_impl);
STARML_REGISTER_KERNEL(square_dispatcher, kCPU, &square_impl);

}  // namespace starml
