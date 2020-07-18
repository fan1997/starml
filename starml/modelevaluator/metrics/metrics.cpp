#include "starml/modelevaluator/metrics/metrics.h"
#include <iostream>

namespace starml {
namespace modelevaluator {
namespace metrics{

float metrics::mean_squared_error(const starml::Matrix& y, const starml::Matrix& y_pred){
  std::cout << "calc mean_squared_error..." << '\n';
  STARML_CHECK_EQ(y.size(), y_pred.size()) << "y and y_pred should have same size";
  float score = -1.0;
  score = mse_score(y, y_pred);
  return score;
}

} // namespace starml
} // namespace modelevaluator
} // namespace metrics
