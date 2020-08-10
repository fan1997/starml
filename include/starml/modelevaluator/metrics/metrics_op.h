#pragma once
#include "starml/basic/matrix.h"
#include "starml/basic/dispatch.h"

namespace starml {
namespace modelevaluator {
namespace metrics{
using common_metric_kernel_fn = float (*)(const Matrix& y,
                               const Matrix& y_pred);

//mse
STARML_DECLARE_DISPATCHER(mse_dispatcher, common_metric_kernel_fn);
float mse_error(const Matrix& y, const Matrix& y_pred);

//acc
STARML_DECLARE_DISPATCHER(acc_dispatcher, common_metric_kernel_fn);
float acc_score(const Matrix& y, const Matrix& y_pred);


} // namespace starml
} // namespace modelevaluator
} // namespace metrics
