#include "index_helper.cuh"
#include "starml/basic/common_cuda.h"
#include "starml/operators/binary_ops.h"
#include "starml/operators/expression.h"

namespace starml {
namespace {
template <typename TScalarType1, typename TScalarType2, typename TResultType,
          typename TOp>
__global__ void binary_kernel(const TScalarType1* data1,
                              const TScalarType2* data2, int start, int end,
                              TOp op, IndexHelper data1_index_helper,
                              IndexHelper data2_index_helper,
                              IndexHelper result_index_helper,
                              TResultType* result_data) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i + start < end) {
    int data1_offset = data1_index_helper.index(i + start);
    int data2_offset = data2_index_helper.index(i + start);
    int result_offset = result_index_helper.index(i + start);
    *(result_data + result_offset) =
        op(*(data1 + data1_offset), *(data2 + data2_offset));
  }
}

template <typename TScalarType1, typename TScalarType2, typename TResultType,
          typename TOp>
void eval_binary(const TScalarType1* data1, const TScalarType2* data2,
                 TResultType* result_data, const Expression& expr, int start,
                 int end, TOp op) {
  int ndims = expr.dims(0).size();
  IndexHelper data1_index_helper =
      IndexHelper(expr.dims(0).data(), expr.strides(0).data(), ndims);
  IndexHelper data2_index_helper =
      IndexHelper(expr.dims(1).data(), expr.strides(1).data(), ndims);
  IndexHelper result_index_helper =
      IndexHelper(expr.dims(2).data(), expr.strides(2).data(), ndims);
  dim3 dimGrid(ceil((end - start) / 256.0), 1, 1);
  dim3 dimBlock(256, 1, 1);
  binary_kernel<<<dimGrid, dimBlock>>>(data1, data2, start, end, op,
                                       data1_index_helper, data2_index_helper,
                                       result_index_helper, result_data);
}

void add_impl(const Matrix& matrix1, const Matrix& matrix2, Matrix& result) {
  auto dtype1 = matrix1.data_type().type();
  auto dtype2 = matrix2.data_type().type();
  auto result_dtype = result.data_type().type();
  Expression expr = Expression(matrix1, matrix2, result);
  STARML_DISPATCH_TYPES(dtype1, "ADD_CUDA", [&]() {
    auto data1 = matrix1.data<scalar_t>();
    using scalar_type1 = scalar_t;
    STARML_DISPATCH_TYPES(dtype2, "ADD_CUDA", [&]() {
      auto data2 = matrix2.data<scalar_t>();
      using scalar_type2 = scalar_t;
      STARML_DISPATCH_TYPES(result_dtype, "ADD_CUDA", [&]() {
        auto result_data = result.mutable_data<scalar_t>();
        using result_scalar_type = scalar_t;
        eval_binary(data1, data2, result_data, expr, 0, result.size(),
                    [=] __device__(scalar_type1 a, scalar_type2 b)
                        -> result_scalar_type { return a + b; });
      });
    });
  });
}

}  // namespace
STARML_REGISTER_KERNEL(add_dispatcher, &add_impl, kCUDA, kCUDA, kCUDA);
// STARML_REGISTER_KERNEL(sub_dispatcher, kCUDA, &sub_impl);
// STARML_REGISTER_KERNEL(mul_dispatcher, kCUDA, &mul_impl);
// STARML_REGISTER_KERNEL(div_dispatcher, kCUDA, &div_impl);
}  // namespace starml