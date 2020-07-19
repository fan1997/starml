#include "starml/modelevaluator/metrics/metrics.h"
#include "starml/preprocessing/scaler/standardscaler.h"
#include <iostream>

namespace starml {
namespace modelevaluator {
namespace metrics{

float metrics::mean_squared_error(const starml::Matrix& y, const starml::Matrix& y_pred){
  std::cout << "calc mean_squared_error..." << '\n';
  STARML_CHECK_EQ(y.size(), y_pred.size()) << "y and y_pred should have same size";
  float err = -1.0;
  err = mse_error(y, y_pred);
  return err;
}

float metrics::r2_score(const starml::Matrix& y, const starml::Matrix& y_pred){
  std::cout << "calc r2_score..." << '\n';
  STARML_CHECK_EQ(y.size(), y_pred.size()) << "y and y_pred should have same size";
  float score = 0.0;
  float mse = -1.0;
  mse = mse_error(y, y_pred);
  starml::preprocessing::scaler::StandardScaler scaler;
  scaler.fit(y);
  Matrix std = scaler.get_std();
  auto data_type = std.data_type().type();
  STARML_DISPATCH_TYPES(data_type, "R2SCORE", [&]() {
      score = 1.0 - mse/(std.data<scalar_t>()[0] * std.data<scalar_t>()[0]);
  });
  return score;

}

} // namespace starml
} // namespace modelevaluator
} // namespace metrics
