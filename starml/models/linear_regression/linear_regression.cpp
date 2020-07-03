#include "starml/models/linear_regression/linear_regression.h"
#include <iostream>

namespace starml {
namespace regression{

regression::LinearRegression::LinearRegression(const starml::Matrix& train_data,
                                        const starml::Matrix& label,
                                        const double lambda) {
  std::cout << "LinearRegression Model has been created with train_data" << "\n";
  this->lambda = lambda;
  printf("train_data:%f",train_data.data()[0]);
  this->train(train_data, label);

}

regression::LinearRegression::LinearRegression(double lambda) {
  this->lambda = lambda;
  std::cout << "LinearRegression Model has been created with lambda = " << lambda << "\n";
}

regression::LinearRegression::LinearRegression(const starml::Matrix& train_data,
                                        const starml::Matrix& label,
                                        const starml::Matrix& weights,
                                        const double lambda) {
  std::cout << "LinearRegression Model has been created with weights" << "\n";
}

double regression::LinearRegression::train(const starml::Matrix& train_data,
                                           const starml::Matrix& label) {
  std::cout << "Training LinearRegression Model" << "\n";
  /**
   * for loop
   * calc w_grad = grad(w)
   * grad_decent : w = w - lr * w_grad
   */
}

double regression::LinearRegression::train(const starml::Matrix& train_data,
                                           const starml::Matrix& label,
                                           const starml::Matrix& weights) {
  std::cout << "Training LinearRegression Model with weights" << "\n";
  /*

  */
}

void regression::LinearRegression::predict(const starml::Matrix& predict_data,
                                           starml::Matrix& predict_result) const {
  std::cout << "Predict LinearRegression Model" << "\n";
}

} // namespace regression
} // namespace starml
