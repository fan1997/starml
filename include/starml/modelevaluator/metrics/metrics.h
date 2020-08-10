#pragma once
#include "starml/basic/matrix.h"
#include "starml/modelevaluator/metrics/metrics_op.h"

namespace starml {
namespace modelevaluator {
namespace metrics{

class metrics {
public:
    metrics() = default;
    float mean_squared_error(const starml::Matrix& y, const starml::Matrix& y_pred);
    float r2_score(const starml::Matrix& y, const starml::Matrix& y_pred);
    float accuracy_score(const starml::Matrix& y, const starml::Matrix& y_pred, bool normalize = true);
};

} // namespace starml
} // namespace modelevaluator
} // namespace metrics
