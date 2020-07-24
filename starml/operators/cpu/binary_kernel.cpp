#include <iostream>

#include "index_helper.h"
#include "starml/operators/binary_ops.h"
#include "starml/operators/expression.h"

namespace starml {
namespace {

template <typename TScalarType1, typename TScalarType2, typename TResultType,
          typename TOp>
void eval_binary(const TScalarType1* data1, const TScalarType2* data2,
                 TResultType* result_data, const Expression& expr, int start,
                 int end, TOp op) {
  IndexHelper data1_index_helper = IndexHelper(expr.dims(0), expr.strides(0));
  IndexHelper data2_index_helper = IndexHelper(expr.dims(1), expr.strides(1));
  IndexHelper result_index_helper = IndexHelper(expr.dims(2), expr.strides(2));
  for (int i = start; i < end; i++) {
    int data1_offset = data1_index_helper.index(i);
    int data2_offset = data2_index_helper.index(i);
    int result_offset = result_index_helper.index(i);
    *(result_data + result_offset) =
        op(*(data1 + data1_offset), *(data2 + data2_offset));
  }
}

void add_impl(const Matrix& matrix1, const Matrix& matrix2, Matrix& result,
              bool blocking) {
  auto dtype1 = matrix1.data_type().type();
  auto dtype2 = matrix2.data_type().type();
  auto result_dtype = result.data_type().type();
  Expression expr = Expression(matrix1, matrix2, result);
  STARML_DISPATCH_TYPES(dtype1, "ADD_CPU", [&]() {
    auto data1 = matrix1.data<scalar_t>();
    using scalar_type1 = scalar_t;
    STARML_DISPATCH_TYPES(dtype2, "ADD_CPU", [&]() {
      auto data2 = matrix2.data<scalar_t>();
      using scalar_type2 = scalar_t;
      STARML_DISPATCH_TYPES(result_dtype, "ADD_CPU", [&]() {
        auto result_data = result.mutable_data<scalar_t>();
        using result_scalar_type = scalar_t;
        eval_binary(data1, data2, result_data, expr, 0, result.size(),
                    [=](scalar_type1 a, scalar_type2 b) -> result_scalar_type {
                      return a + b;
                    });
      });
    });
  });
}

void sub_impl(const Matrix& matrix1, const Matrix& matrix2, Matrix& result,
              bool blocking) {
  auto dtype1 = matrix1.data_type().type();
  auto dtype2 = matrix2.data_type().type();
  auto result_dtype = result.data_type().type();
  Expression expr = Expression(matrix1, matrix2, result);
  STARML_DISPATCH_TYPES(dtype1, "SUB_CPU", [&]() {
    auto data1 = matrix1.data<scalar_t>();
    using scalar_type1 = scalar_t;
    STARML_DISPATCH_TYPES(dtype2, "SUB_CPU", [&]() {
      auto data2 = matrix2.data<scalar_t>();
      using scalar_type2 = scalar_t;
      STARML_DISPATCH_TYPES(result_dtype, "SUB_CPU", [&]() {
        auto result_data = result.mutable_data<scalar_t>();
        using result_scalar_type = scalar_t;
        eval_binary(data1, data2, result_data, expr, 0, result.size(),
                    [=](scalar_type1 a, scalar_type2 b) -> result_scalar_type {
                      return a - b;
                    });
      });
    });
  });
}

void mul_impl(const Matrix& matrix1, const Matrix& matrix2, Matrix& result,
              bool blocking) {
  auto dtype1 = matrix1.data_type().type();
  auto dtype2 = matrix2.data_type().type();
  auto result_dtype = result.data_type().type();
  Expression expr = Expression(matrix1, matrix2, result);
  STARML_DISPATCH_TYPES(dtype1, "MUL_CPU", [&]() {
    auto data1 = matrix1.data<scalar_t>();
    using scalar_type1 = scalar_t;
    STARML_DISPATCH_TYPES(dtype2, "MUL_CPU", [&]() {
      auto data2 = matrix2.data<scalar_t>();
      using scalar_type2 = scalar_t;
      STARML_DISPATCH_TYPES(result_dtype, "MUL_CPU", [&]() {
        auto result_data = result.mutable_data<scalar_t>();
        using result_scalar_type = scalar_t;
        eval_binary(data1, data2, result_data, expr, 0, result.size(),
                    [=](scalar_type1 a, scalar_type2 b) -> result_scalar_type {
                      return a * b;
                    });
      });
    });
  });
}

void div_impl(const Matrix& matrix1, const Matrix& matrix2, Matrix& result,
              bool blocking) {
  auto dtype1 = matrix1.data_type().type();
  auto dtype2 = matrix2.data_type().type();
  auto result_dtype = result.data_type().type();
  Expression expr = Expression(matrix1, matrix2, result);
  STARML_DISPATCH_TYPES(dtype1, "DIV_CPU", [&]() {
    auto data1 = matrix1.data<scalar_t>();
    using scalar_type1 = scalar_t;
    STARML_DISPATCH_TYPES(dtype2, "DIV_CPU", [&]() {
      auto data2 = matrix2.data<scalar_t>();
      using scalar_type2 = scalar_t;
      STARML_DISPATCH_TYPES(result_dtype, "DIV_CPU", [&]() {
        auto result_data = result.mutable_data<scalar_t>();
        using result_scalar_type = scalar_t;
        eval_binary(data1, data2, result_data, expr, 0, result.size(),
                    [=](scalar_type1 a, scalar_type2 b) -> result_scalar_type {
                      return a / b;
                    });
      });
    });
  });
}

}  // namespace

STARML_REGISTER_KERNEL(add_dispatcher, kCPU, &add_impl);
STARML_REGISTER_KERNEL(sub_dispatcher, kCPU, &sub_impl);
STARML_REGISTER_KERNEL(mul_dispatcher, kCPU, &mul_impl);
STARML_REGISTER_KERNEL(div_dispatcher, kCPU, &div_impl);

}  // namespace starml