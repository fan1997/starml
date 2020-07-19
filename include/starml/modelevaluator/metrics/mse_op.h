#pragma once
#include "starml/basic/matrix.h"
#include "starml/basic/dispatch.h"

namespace starml {
namespace modelevaluator {
namespace metrics{

using mse_kernel_fn = float (*)(const Matrix& y,
                               const Matrix& y_pred);
STARML_DECLARE_DISPATCHER(mse_dispatcher, mse_kernel_fn);

float mse_error(const Matrix& y, const Matrix& y_pred);

} // namespace starml
} // namespace modelevaluator
} // namespace metrics
