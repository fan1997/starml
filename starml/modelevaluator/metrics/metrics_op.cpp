#include "starml/modelevaluator/metrics/metrics_op.h"

namespace starml {
namespace modelevaluator {
namespace metrics{
  STARML_DEFINE_DISPATCHER(mse_dispatcher);
  STARML_DEFINE_DISPATCHER(acc_dispatcher);
  
//mse
  float mse_error(const Matrix& y, const Matrix& y_pred) {
    float score = 0.0;
    score = mse_dispatcher(y, y_pred);
    return score;
  }
//accuracy
  float acc_score(const Matrix& y, const Matrix& y_pred) {
    float score = 0.0;
    score = acc_dispatcher(y, y_pred);
    return score;
  }

} //
} //
} //
