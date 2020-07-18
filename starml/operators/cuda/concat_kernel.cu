#include "starml/operators/concat.h"
#include "starml/basic/common_cuda.h"

namespace starml {
namespace {
template <typename T>
__global__ void concat_kernel(T* data_1, T* data_2, T* res_data, int size, int w1, int w2) {
  int pos = blockDim.x * blockIdx.x + threadIdx.x;
  if (pos < size) {
      int n = pos / (w1 + w2);
      int m = pos % (w1 + w2);
      res_data[pos] = m >= w1 ? data_2[n * w2 + m - w1] : data_1[n * w1 + m];
  }
}

void concat_impl(const Matrix& matrix1, const Matrix& matrix2, Matrix& result, int axis) {
// void concat_impl(const Matrix& matrix1, const Matrix& matrix2, Matrix& result) {
  // int axis = 1; // temp
  auto m1_rows_num =  matrix1.dim(0);
  auto m1_cols_num =  matrix1.dim(1);
  auto m2_rows_num =  matrix2.dim(0);
  auto m2_cols_num =  matrix2.dim(1);
  auto w1 = axis == 0 ? m1_rows_num : m1_cols_num;
  auto w2 = axis == 0 ? m2_rows_num : m2_cols_num;
  auto data_type = result.data_type().type();
  auto size = result.size();
  STARML_DISPATCH_TYPES(data_type, "CONCAT", [&]() {
    scalar_t *data_1 = matrix1.data<scalar_t>();
    scalar_t *data_2 = matrix2.data<scalar_t>();
    scalar_t *res_data = result.data<scalar_t>();
    dim3 dimGrid(ceil(size / 256.0), 1, 1);
    dim3 dimBlock(256, 1, 1);
    concat_kernel<scalar_t><<<dimGrid, dimBlock>>>(data_1, data_2, res_data, size, w1, w2);
  });
}
}  // namespace
STARML_REGISTER_KERNEL(concat_dispatcher, kCUDA, &concat_impl);

}  // namespace starml
