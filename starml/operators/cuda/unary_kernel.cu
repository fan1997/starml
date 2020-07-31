#include "starml/basic/common_cuda.h"
#include "starml/operators/unary_ops.h"

namespace starml {
namespace {
template <typename TScalarType, typename TResultType, typename TOp>
__global__ void unary_kernel(const TScalarType* data, int start, int end,
                             TOp op, TResultType* result_data) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i + start < end) {
    *(result_data + i + start) = op(*(data + i + start));
  }
}

template <typename TScalarType, typename TResultType, typename TOp>
void eval_unary(const TScalarType* data, TResultType* result_data, int start,
                int end, TOp op) {
  dim3 dimGrid(ceil((end - start) / 256.0), 1, 1);
  dim3 dimBlock(256, 1, 1);
  unary_kernel<<<dimGrid, dimBlock>>>(data, start, end, op, result_data);
}

void exp_impl(const Matrix& matrix, Matrix& result) {
  auto dtype = matrix.data_type().type();
  auto result_dtype = result.data_type().type();
  auto cast_dtype = (dtype < result_dtype) ? result_dtype : dtype;
  STARML_DISPATCH_TYPES(result_dtype, "CUDA_EXP", [&]() {
    auto result_data = result.mutable_data<scalar_t>();
    using result_scalar_type = scalar_t;
    STARML_DISPATCH_TYPES(dtype, "CUDA_EXP", [&]() {
      auto data = matrix.data<scalar_t>();
      using scalar_type = scalar_t;
      STARML_DISPATCH_FLOATING_TYPES(cast_dtype, "CUDA_EXP", [&]() {
        eval_unary(data, result_data, 0, result.size(),
                   [=] __device__(scalar_type a) -> result_scalar_type {
                     return ::exp(scalar_t(a));
                   });
      });
    });
  });
}

void log_impl(const Matrix& matrix, Matrix& result) {
  auto dtype = matrix.data_type().type();
  auto result_dtype = result.data_type().type();
  auto cast_dtype = (dtype < result_dtype) ? result_dtype : dtype;
  STARML_DISPATCH_TYPES(result_dtype, "CUDA_LOG", [&]() {
    auto result_data = result.mutable_data<scalar_t>();
    using result_scalar_type = scalar_t;
    STARML_DISPATCH_TYPES(dtype, "CUDA_LOG", [&]() {
      auto data = matrix.data<scalar_t>();
      using scalar_type = scalar_t;
      STARML_DISPATCH_FLOATING_TYPES(cast_dtype, "CUDA_LOG", [&]() {
        eval_unary(data, result_data, 0, result.size(),
                   [=] __device__(scalar_type a) -> result_scalar_type {
                     return ::log(scalar_t(a));
                   });
      });
    });
  });
}

void negtive_impl(const Matrix& matrix, Matrix& result) {
  auto dtype = matrix.data_type().type();
  auto result_dtype = result.data_type().type();
  auto cast_dtype = (dtype < result_dtype) ? result_dtype : dtype;
  STARML_DISPATCH_TYPES(result_dtype, "CUDA_NEG", [&]() {
    auto result_data = result.mutable_data<scalar_t>();
    using result_scalar_type = scalar_t;
    STARML_DISPATCH_TYPES(dtype, "CUDA_NEG", [&]() {
      auto data = matrix.data<scalar_t>();
      using scalar_type = scalar_t;
      STARML_DISPATCH_FLOATING_TYPES(cast_dtype, "CUDA_NEG", [&]() {
        eval_unary(data, result_data, 0, result.size(),
                   [=] __device__(scalar_type a) -> result_scalar_type { return - a;});
      });
    });
  });
}

}  // namespace
STARML_REGISTER_KERNEL(exp_dispatcher, &exp_impl, kCUDA, kCUDA);
STARML_REGISTER_KERNEL(log_dispatcher, &log_impl, kCUDA, kCUDA);
STARML_REGISTER_KERNEL(negtive_dispatcher, &negtive_impl, kCUDA, kCUDA);
}  // namespace starml
