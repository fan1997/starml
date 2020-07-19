#include "starml/models/linear_regression/linear_regression.h"
#include "starml/operators/transpose.h"
#include "starml/operators/add_scalar.h"
#include "starml/operators/solve.h"
#include "starml/operators/matmul.h"
#include <iostream>

namespace starml {
namespace models{
namespace regression{

regression::LinearRegression::LinearRegression(const starml::Matrix& train_data,
                                        const starml::Matrix& label,
                                        const float lambda) {
  std::cout << "LinearRegression Model has been created with train_data" << "\n";
  this->lambda = lambda;
  this->train(train_data, label);

}

regression::LinearRegression::LinearRegression(float lambda) {
  this->lambda = lambda;
  std::cout << "LinearRegression Model has been created with lambda = " << lambda << "\n";
}

regression::LinearRegression::LinearRegression(const starml::Matrix& train_data,
                                        const starml::Matrix& label,
                                        const starml::Matrix& weights,
                                        const float lambda) {
  std::cout << "LinearRegression Model has been created with weights" << "\n";
}

float regression::LinearRegression::train(const starml::Matrix& train_data,
                                           const starml::Matrix& label) {
  std::cout << "Training LinearRegression Model" << "\n";
  STARML_CHECK_EQ(train_data.dim(0), label.dim(0)) << "train_data and label should have same rows";
  /**
   * (XT*X + lambda * i) W = XT*Y
   * add AX = B -> X solver in cpu(EIGEN) && cuda (CUSOLVER)
   * 1.XT = TRANSPOSE(X)
   * 2.XTX = matmul(XT, X)
   * 3.XTY = matmul(XT, Y)
   * 4.W = linearsolver
   */
   starml::Matrix train_data_t = transpose(train_data);
   starml::Matrix xtx = matmul(train_data_t, train_data);
   starml::Matrix xty = matmul(train_data_t, label);
   this -> parameters = lu_solve(add_scalar(xtx, this -> lambda), xty);
}

float regression::LinearRegression::train(const starml::Matrix& train_data,
                                           const starml::Matrix& label,
                                           const starml::Matrix& weights) {
  std::cout << "Training LinearRegression Model with weights" << "\n";
  /*

  */
}

Matrix& regression::LinearRegression::predict(const starml::Matrix& predict_data,
                                           starml::Matrix& predict_result) const {
  std::cout << "Predict LinearRegression Model" << "\n";
  STARML_CHECK_EQ(predict_data.dim(1), this->parameters.dim(0)) << "The predict_data's features num should be the same as model weights ";
  predict_result = matmul(predict_data, this->parameters);
  return predict_result;
}

Matrix regression::LinearRegression::predict(const starml::Matrix& predict_data) const {
  std::cout << "Predict LinearRegression Model" << "\n";
  STARML_CHECK_EQ(predict_data.dim(1), this->parameters.dim(0)) << "The predict_data's features num should be the same as model weights ";
  Matrix predict_result = matmul(predict_data, this->parameters);
  return predict_result;
}

} // namespace regression
} // models
} // namespace starml
