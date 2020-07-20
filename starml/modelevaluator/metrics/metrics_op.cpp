#include "starml/modelevaluator/metrics/metrics_op.h"

namespace starml {
namespace modelevaluator {
namespace metrics{
//mse
  STARML_DEFINE_DISPATCHER(mse_dispatcher);
  float mse_error(const Matrix& y, const Matrix& y_pred) {
    float score = 0.0;
    score = mse_dispatcher(y, y_pred);
    return score;
  }
//accuracy
  STARML_DEFINE_DISPATCHER(acc_dispatcher);
  float acc_score(const Matrix& y, const Matrix& y_pred) {
    float score = 0.0;
    score = acc_dispatcher(y, y_pred);
    return score;
  }

} //
} //
} //
