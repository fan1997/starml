#include "starml/operators/add_scalar.h"
#include "starml/basic/common_cuda.h"

namespace starml {
namespace {
template <typename T>
__global__ void add_scalar_impl_kernel(T* data_1, T* res_data, double b, int size){
   int i = blockDim.x * blockIdx.x + threadIdx.x;
   if (i < size) {
      res_data[i] = data_1[i] + b;
   }
}
void add_scalar_impl(const Matrix& matrix1,  Matrix& result, double b) {
  auto data_type = matrix1.data_type().type();
  int size = matrix1.size();
  STARML_DISPATCH_TYPES(data_type, "ADDSCALAR", [&]() {
    auto data_1 = matrix1.data<scalar_t>();
    auto res_data = result.data<scalar_t>();
    dim3 dimGrid(ceil(size / 256.0), 1, 1);
    dim3 dimBlock(256, 1, 1);
    add_scalar_impl_kernel<scalar_t><<<dimGrid, dimBlock>>>(data_1, res_data, b, size);
    cudaDeviceSynchronize();
  });
}

}  // namespace

STARML_REGISTER_KERNEL(add_scalar_dispatcher, kCUDA, &add_scalar_impl);

}  // namespace starml
