#include "starml/modelevaluator/metrics/mse_op.h"

namespace starml {
namespace modelevaluator {
namespace metrics{
  STARML_DEFINE_DISPATCHER(mse_dispatcher);
  float mse_score(const Matrix& y, const Matrix& y_pred) {
    float score = 0.0;
    score = mse_dispatcher(y, y_pred);
    return score;
  }
}
}
}
