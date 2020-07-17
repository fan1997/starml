#include "starml/operators/binary_ops.h"
#include "starml/basic/common_cuda.h"

namespace starml {
namespace {
template <typename T>
__global__ void add_kernel(const T* a, const T* b, int nums, T* res) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < nums) {
    res[i] = a[i] + b[i];
  }
}

void add_impl(const Matrix& matrix1, const Matrix& matrix2, Matrix& result) {
  auto data_type = matrix1.data_type().type();
  int nums = matrix1.size();
  STARML_DISPATCH_TYPES(data_type, "ADD", [&]() {
    scalar_t *data1 = matrix1.data<scalar_t>();
    scalar_t *data2 = matrix2.data<scalar_t>();
    scalar_t *res_data = result.data<scalar_t>();
    dim3 dimGrid(ceil(nums / 256.0), 1, 1);
    dim3 dimBlock(256, 1, 1);
    add_kernel<scalar_t><<<dimGrid, dimBlock>>>(data1, data2, nums, res_data);
  });
}

template <typename T>
__global__ void sub_kernel(const T* a, const T* b, int nums, T* res) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < nums) {
    res[i] = a[i] - b[i];
  }
}

void sub_impl(const Matrix& matrix1, const Matrix& matrix2, Matrix& result) {
  auto data_type = matrix1.data_type().type();
  int nums = matrix1.rows_num() * matrix1.cols_num();
  STARML_DISPATCH_TYPES(data_type, "SUB", [&]() {
    scalar_t *data1 = matrix1.data<scalar_t>();
    scalar_t *data2 = matrix2.data<scalar_t>();
    scalar_t *res_data = result.data<scalar_t>();
    dim3 dimGrid(ceil(nums / 256.0), 1, 1);
    dim3 dimBlock(256, 1, 1);
    sub_kernel<scalar_t><<<dimGrid, dimBlock>>>(data1, data2, nums, res_data);
  });
}
}  // namespace
STARML_REGISTER_KERNEL(add_dispatcher, kCUDA, &add_impl);
STARML_REGISTER_KERNEL(sub_dispatcher, kCUDA, &sub_impl);
}  // namespace starml
