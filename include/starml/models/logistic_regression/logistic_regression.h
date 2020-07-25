#pragma once
#include "starml/basic/matrix.h"
#include "starml/optimizer/optimizer.h"

namespace starml {
namespace models {
namespace classification {

struct LogisticRegressionParam {
    LogisticRegressionParam() {
        solver_type = starml::optimizer::kSGD;
        learning_rate = 0.001;
        lambda = 0;
    }
    float lambda;
    float learning_rate;
    starml::optimizer::SolverType solver_type;
};

class LogisticRegression
{
 public:

  LogisticRegression(const starml::Matrix& train_data,
                     const starml::Matrix& label,
                     const float lambda = 0.0);

  LogisticRegression(const starml::Matrix& train_data,
                   const starml::Matrix& label,
                   const starml::Matrix& weights,
                   const float lambda = 0.0);

  LogisticRegression(float lambda = 0.0);
  LogisticRegression(LogisticRegressionParam& param);
  float train(const starml::Matrix& train_data,
              const starml::Matrix& label);

  /**
   * Train the LinearRegression model on the given data and weights. Careful!
   * This will completely ignore and overwrite the existing model. This
   * particular implementation does not have an incremental training algorithm.
   * To set the regularization parameter lambda, call Lambda() or set a
   * different value in the constructor.
   *
   * @param predictors X, the matrix of data points to train the model on.
   * @param responses y, the responses to the data points.
   * @param weights Observation weights (for boosting).
   * @param intercept Whether or not to fit an intercept term.
   * @return The least squares error after training.
   */
  float train(const starml::Matrix& train_data,
               const starml::Matrix& label,
               const starml::Matrix& weights);

  /**
   * Calculate y_i for each data point in points.
   *
   * @param predict_data the data points to calculate with.
   * @param predictions y, will contain calculated values on completion.
   */
  Matrix& predict(const starml::Matrix& predict_data, starml::Matrix& predict_result) const;
  Matrix predict(const starml::Matrix& predict_data) const;
  //! Return the parameters (the b vector).
  const starml::Matrix& get_parameters() const { return parameters; }
  //! Modify the parameters (the b vector).
  starml::Matrix& get_parameters() { return parameters; }

  //! Return the parameters (the b vector).
  const LogisticRegressionParam& get_param() const { return param; }
  //! Modify the parameters (the b vector).
  LogisticRegressionParam& get_param() { return param; }

 private:

  starml::Matrix parameters;
  std::shared_ptr<starml::optimizer::Optimizer> optimizer;
  LogisticRegressionParam param;

};

} // namespace classification
} // models
} // namespace starml
