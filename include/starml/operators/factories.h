#pragma once
#include "starml/basic/matrix.h"
#include "starml/basic/scalar.h"

namespace starml {
Matrix full(const Shape& shape, const Device& device, const DataType& data_type,
            const Scalar& init_val);
Matrix empty(const Shape& shape, const Device& device,
             const DataType& data_type);
Matrix cast(const Matrix& matrix, const DataType& data_type);
}  // namespace starml
