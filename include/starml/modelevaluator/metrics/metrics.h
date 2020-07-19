#pragma once
#include "starml/basic/matrix.h"
#include "starml/modelevaluator/metrics/mse_op.h"

namespace starml {
namespace modelevaluator {
namespace metrics{

class metrics {

public:
    metrics() = default;
    float mean_squared_error(const starml::Matrix& y, const starml::Matrix& y_pred);
    float r2_score(const starml::Matrix& y, const starml::Matrix& y_pred);
};

} // namespace starml
} // namespace modelevaluator
} // namespace metrics
