#pragma once
#include "starml/basic/matrix.h"
#include "starml/basic/scalar.h"
#include "starml/basic/dispatch.h"

namespace starml {
using full_kernel_fn = void (*)(const Scalar& init_val, Matrix& result);
STARML_DECLARE_DISPATCHER(full_dispatcher, full_kernel_fn);

Matrix full(const Shape& shape, const Device& device, const DataType& data_type,
            const Scalar& init_val);
Matrix empty(const Shape& shape, const Device& device,
             const DataType& data_type);
}  // namespace starml
