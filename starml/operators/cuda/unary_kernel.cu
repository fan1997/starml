#include "starml/operators/unary_ops.h"
#include "starml/basic/common_cuda.h"
#include "starml/basic/context_cuda.h"

namespace starml {
namespace {
template <typename TScalarType, typename TOp>
__global__ void unary_kernel(const TScalarType* data, int start, int end,
                             TOp op, TScalarType* result_data) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i + start < end) {
    *(result_data + i + start) = op(*(data + i + start));
  }
}

template <typename TScalarType, typename TOp>
void eval_unary(const TScalarType* data, TScalarType* result_data, int start,
                int end, CUDAContext* cuda_ctx, TOp op) {
  dim3 dimGrid(ceil((end - start) / 256.0), 1, 1);
  dim3 dimBlock(256, 1, 1);
  unary_kernel<TScalarType><<<dimGrid, dimBlock, 0, cuda_ctx->stream()>>>(
      data, start, end, op, result_data);
}

void exp_impl(const Matrix& matrix, Matrix& result) {
  auto data_type = matrix.data_type().type();
  auto cuda_ctx = get_cuda_context(matrix.device());
  STARML_DISPATCH_FLOATING_TYPES(data_type, "CUDA_EXP", [&]() {
    auto data = matrix.data<scalar_t>();
    auto result_data = result.mutable_data<scalar_t>();
    cuda_ctx->prefetch_async(const_cast<scalar_t*>(data),
                             sizeof(scalar_t) * matrix.size(),
                             cuda_ctx->stream());
    cuda_ctx->prefetch_async(result_data, sizeof(scalar_t) * result.size(),
                             cuda_ctx->stream());
    eval_unary<scalar_t>(
        data, result_data, 0, result.size(), cuda_ctx,
        [=] __device__(scalar_t a) -> scalar_t { return ::exp(a); });
    cuda_ctx->synchronize();
  });
}

}  // namespace
STARML_REGISTER_KERNEL(exp_dispatcher, kCUDA, &exp_impl);
}  // namespace starml