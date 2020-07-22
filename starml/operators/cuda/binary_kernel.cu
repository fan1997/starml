#include "starml/operators/binary_ops.h"
#include "starml/basic/common_cuda.h"
#include "starml/basic/context_cuda.h"
#include "starml/operators/expression.h"
#include "index_helper.cuh"

namespace starml {
namespace {
template <typename TScalarType, typename TOp>
__global__ void binary_kernel(const TScalarType* data1,
                              const TScalarType* data2, int start, int end,
                              TOp op, IndexHelper data1_index_helper,
                              IndexHelper data2_index_helper,
                              IndexHelper result_index_helper,
                              TScalarType* result_data) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i + start < end) {
    int data1_offset = data1_index_helper.index(i + start);
    int data2_offset = data2_index_helper.index(i + start);
    int result_offset = result_index_helper.index(i + start);
    *(result_data + result_offset) =
        op(*(data1 + data1_offset), *(data2 + data2_offset));
  }
}

template <typename TScalarType, typename TOp>
void eval_binary(const TScalarType* data1, const TScalarType* data2,
                 TScalarType* result_data, const Expression& expr, int start,
                 int end, CUDAContext* cuda_ctx, TOp op) {
  int ndims = expr.dims(0).size();
  IndexHelper data1_index_helper =
      IndexHelper(expr.dims(0).data(), expr.strides(0).data(), ndims);
  IndexHelper data2_index_helper =
      IndexHelper(expr.dims(1).data(), expr.strides(1).data(), ndims);
  IndexHelper result_index_helper =
      IndexHelper(expr.dims(2).data(), expr.strides(2).data(), ndims);
  dim3 dimGrid(ceil((end - start) / 256.0), 1, 1);
  dim3 dimBlock(256, 1, 1);
  binary_kernel<TScalarType><<<dimGrid, dimBlock, 0, cuda_ctx->stream()>>>(
      data1, data2, start, end, op, data1_index_helper, data2_index_helper,
      result_index_helper, result_data);
}

void add_impl(const Matrix& matrix1, const Matrix& matrix2, Matrix& result) {
  auto data_type = matrix1.data_type().type();
  Expression expr = Expression(matrix1, matrix2, result);
  auto cuda_ctx = get_cuda_context(matrix1.device());
  STARML_DISPATCH_TYPES(data_type, "CUDA_ADD", [&]() {
    auto data1 = matrix1.data<scalar_t>();
    auto data2 = matrix2.data<scalar_t>();
    auto result_data = result.mutable_data<scalar_t>();
    cuda_ctx->prefetch_async(const_cast<scalar_t*>(data1),
                             sizeof(scalar_t) * matrix1.size(),
                             cuda_ctx->stream());
    cuda_ctx->prefetch_async(const_cast<scalar_t*>(data2),
                             sizeof(scalar_t) * matrix2.size(),
                             cuda_ctx->stream());
    cuda_ctx->prefetch_async(result_data, sizeof(scalar_t) * result.size(),
                             cuda_ctx->stream());
    eval_binary<scalar_t>(
        data1, data2, result_data, expr, 0, result.size(), cuda_ctx,
        [=]__device__ (scalar_t a, scalar_t b) -> scalar_t { return a + b; });
    cuda_ctx->synchronize();
  });
}

void sub_impl(const Matrix& matrix1, const Matrix& matrix2, Matrix& result) {
  auto data_type = matrix1.data_type().type();
  Expression expr = Expression(matrix1, matrix2, result);
  auto cuda_ctx = get_cuda_context(matrix1.device());
  STARML_DISPATCH_TYPES(data_type, "CUDA_SUB", [&]() {
    auto data1 = matrix1.data<scalar_t>();
    auto data2 = matrix2.data<scalar_t>();
    auto result_data = result.mutable_data<scalar_t>();
    cuda_ctx->prefetch_async(const_cast<scalar_t*>(data1),
                             sizeof(scalar_t) * matrix1.size(),
                             cuda_ctx->stream());
    cuda_ctx->prefetch_async(const_cast<scalar_t*>(data2),
                             sizeof(scalar_t) * matrix2.size(),
                             cuda_ctx->stream());
    cuda_ctx->prefetch_async(result_data, sizeof(scalar_t) * result.size(),
                             cuda_ctx->stream());
    eval_binary<scalar_t>(
        data1, data2, result_data, expr, 0, result.size(), cuda_ctx,
        [=]__device__ (scalar_t a, scalar_t b) -> scalar_t { return a - b; });
    cuda_ctx->synchronize();
  });
}

void mul_impl(const Matrix& matrix1, const Matrix& matrix2, Matrix& result) {
  auto data_type = matrix1.data_type().type();
  Expression expr = Expression(matrix1, matrix2, result);
  auto cuda_ctx = get_cuda_context(matrix1.device());
  STARML_DISPATCH_TYPES(data_type, "CUDA_MUL", [&]() {
    auto data1 = matrix1.data<scalar_t>();
    auto data2 = matrix2.data<scalar_t>();
    auto result_data = result.mutable_data<scalar_t>();
    cuda_ctx->prefetch_async(const_cast<scalar_t*>(data1),
                             sizeof(scalar_t) * matrix1.size(),
                             cuda_ctx->stream());
    cuda_ctx->prefetch_async(const_cast<scalar_t*>(data2),
                             sizeof(scalar_t) * matrix2.size(),
                             cuda_ctx->stream());
    cuda_ctx->prefetch_async(result_data, sizeof(scalar_t) * result.size(),
                             cuda_ctx->stream());
    eval_binary<scalar_t>(
        data1, data2, result_data, expr, 0, result.size(), cuda_ctx,
        [=]__device__ (scalar_t a, scalar_t b) -> scalar_t { return a * b; });
    cuda_ctx->synchronize();
  });
}

void div_impl(const Matrix& matrix1, const Matrix& matrix2, Matrix& result) {
  auto data_type = matrix1.data_type().type();
  Expression expr = Expression(matrix1, matrix2, result);
  auto cuda_ctx = get_cuda_context(matrix1.device());
  STARML_DISPATCH_TYPES(data_type, "CUDA_DIV", [&]() {
    auto data1 = matrix1.data<scalar_t>();
    auto data2 = matrix2.data<scalar_t>();
    auto result_data = result.mutable_data<scalar_t>();
    cuda_ctx->prefetch_async(const_cast<scalar_t*>(data1),
                             sizeof(scalar_t) * matrix1.size(),
                             cuda_ctx->stream());
    cuda_ctx->prefetch_async(const_cast<scalar_t*>(data2),
                             sizeof(scalar_t) * matrix2.size(),
                             cuda_ctx->stream());
    cuda_ctx->prefetch_async(result_data, sizeof(scalar_t) * result.size(),
                             cuda_ctx->stream());
    eval_binary<scalar_t>(
        data1, data2, result_data, expr, 0, result.size(), cuda_ctx,
        [=]__device__ (scalar_t a, scalar_t b) -> scalar_t { return a / b; });
    cuda_ctx->synchronize();
  });
}

}  // namespace
STARML_REGISTER_KERNEL(add_dispatcher, kCUDA, &add_impl);
STARML_REGISTER_KERNEL(sub_dispatcher, kCUDA, &sub_impl);
STARML_REGISTER_KERNEL(mul_dispatcher, kCUDA, &mul_impl);
STARML_REGISTER_KERNEL(div_dispatcher, kCUDA, &div_impl);
}  // namespace starml