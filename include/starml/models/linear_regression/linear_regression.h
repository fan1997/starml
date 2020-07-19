
#pragma once
#include "starml/basic/matrix.h"

namespace starml {
namespace models {
namespace regression {

/**
 * linear regression class using least squares loss.
 * Optionally, this class can perform ridge regression, if the lambda parameter
 * is set to a number greater than zero.
 */
class LinearRegression
{
 public:
  /**
   * Creates the model.
   *
   * @param train_data X, matrix of data points.
   * @param label y, the measured data for each point in X.
   * @param lambda Regularization constant for ridge regression.
   */
  LinearRegression(const starml::Matrix& train_data,
                   const starml::Matrix& label,
                   const float lambda = 0.0);

  /**
   * Creates the model with weighted learning.
   *
   * @param predictors X, matrix of data points.
   * @param responses y, the measured data for each point in X.
   * @param weights Observation weights (for boosting).
   * @param lambda Regularization constant for ridge regression.
   * @param intercept Whether or not to include an intercept term.
   */
  LinearRegression(const starml::Matrix& train_data,
                   const starml::Matrix& label,
                   const starml::Matrix& weights,
                   const float lambda = 0.0);


  /**
   * Empty constructor.  This gives a non-working model, so make sure Train() is
   * called (or make sure the model parameters are set) before calling
   * Predict()!
   */
  LinearRegression(float lambda = 0.0);
  /**
   * Train the LinearRegression model on the given data. Careful! This will
   * completely ignore and overwrite the existing model. This particular
   * implementation does not have an incremental training algorithm.  To set the
   * regularization parameter lambda, call Lambda() or set a different value in
   * the constructor.
   *
   * @param predictors X, the matrix of data points to train the model on.
   * @param responses y, the responses to the data points.
   * @param intercept Whether or not to fit an intercept term.
   * @return The least squares error after training.
   */
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

  //! Return the Tikhonov regularization parameter for ridge regression.
  float get_lambda() const { return lambda; }
  //! Modify the Tikhonov regularization parameter for ridge regression.
  float& get_lambda() { return lambda; }

 private:
  /**
   * The calculated w.
   * Initialized and filled by constructor to hold the least squares solution.
   */
  starml::Matrix parameters;
  /**
   * The Tikhonov regularization parameter for ridge regression (0 for linear
   * regression).
   */
  float lambda;
};

} // namespace regression
} // models
} // namespace starml
