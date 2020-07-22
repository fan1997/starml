#include "starml/operators/factories.h"
#include "starml/basic/dispatch.h"
namespace starml {
Matrix full(const Shape& shape, const Device& device, const DataType& data_type,
            const Scalar& init_val) {
  Matrix res(shape, device, data_type);
  int size = 1;
  for (auto item : shape) {
    size *= item;
  }
  STARML_DISPATCH_TYPES(data_type.type(), "full", [&]() {
    auto data = res.mutable_data<scalar_t>();
    scalar_t value = init_val.value<scalar_t>();
    for (int i = 0; i < size; i++) {
      data[i] = value;
    }
  });
  return res;
}

Matrix empty(const Shape& shape, const Device& device,
             const DataType& data_type) {
  Matrix res(shape, device, data_type);
  return res;
}

template <typename TScalarType>
void cast_helper(const TScalarType* data, Matrix& result) {
  int size = result.size();
  STARML_DISPATCH_TYPES(result.data_type().type(), "cast", [&]() {
    auto result_data = result.mutable_data<scalar_t>();
    for (int i = 0; i < size; i++) {
      result_data[i] = static_cast<scalar_t>(data[i]);
    }
  });
}

Matrix cast(const Matrix& matrix, const DataType& data_type) {
  Matrix result = Matrix(matrix.dims(), matrix.device(), data_type);
  STARML_DISPATCH_TYPES(matrix.data_type().type(), "cast", [&]() {
    auto data = matrix.data<scalar_t>();
    cast_helper<scalar_t>(data, result);
  }); 
  return result;
}
}  // namespace starml