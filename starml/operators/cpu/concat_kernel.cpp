#include "starml/operators/concat.h"
#include <omp.h>

namespace starml {
namespace {
template <typename T>
void concat_impl_kernel_v1(const T* data_1, const T* data_2, T* res_data, int& size, int& w1, int& w2){
#pragma omp parallel for
  for (int pos = 0; pos < size; pos ++) {
    int n = pos / (w1 + w2);
    int m = pos % (w1 + w2);
    res_data[pos] = m >= w1 ? data_2[n * w2 + m - w1] : data_1[n * w1 + m];
  }
}
template <typename T>
void concat_impl_kernel_v2(const T* data_1, const T* data_2, T* res_data, int& size, int& w1, int& w2, int& cols_num){
#pragma omp parallel for
  for (int pos = 0; pos < size; pos ++) {
    int n = pos % cols_num;
    int m = pos / cols_num;
    res_data[pos] = m >= w1 ? data_2[(m - w1) * cols_num + n] : data_1[m * cols_num + n];
  }
}
void concat_impl(const Matrix& matrix1, const Matrix& matrix2, Matrix& result, int axis) {
// void concat_impl(const Matrix& matrix1, const Matrix& matrix2, Matrix& result) {
  // int axis = 1;
  // std::cout << "In concat_impl " << std::endl;
  auto m1_rows_num =  matrix1.dim(0);
  auto m1_cols_num =  matrix1.dim(1);
  auto m2_rows_num =  matrix2.dim(0);
  auto m2_cols_num =  matrix2.dim(1);
  auto w1 = axis == 0 ? m1_rows_num : m1_cols_num;
  auto w2 = axis == 0 ? m2_rows_num : m2_cols_num;
  auto data_type = result.data_type().type();
  int size = result.size();
  STARML_DISPATCH_TYPES(data_type, "CONCAT", [&]() {
    auto data_1 = matrix1.data<scalar_t>();
    auto data_2 = matrix2.data<scalar_t>();
    auto res_data = result.mutable_data<scalar_t>();
    axis == 1 ? concat_impl_kernel_v1(data_1, data_2, res_data, size, w1, w2) :
    concat_impl_kernel_v2(data_1, data_2, res_data, size, w1, w2, m2_cols_num);
  });
}
}  // namespace

STARML_REGISTER_KERNEL(concat_dispatcher, kCPU, &concat_impl);
}  // namespace starml
