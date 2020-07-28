#include "starml/models/logistic_regression/logistic_regression.h"
#include "starml/operators/binary_ops.h"
#include "starml/operators/unary_ops.h"
#include "starml/optimizer/optimizer.h"
#include "starml/optimizer/SGD.h"
#include "starml/operators/transpose.h"
#include "starml/operators/solve.h"
#include "starml/operators/matmul.h"
#include "starml/operators/factories.h"
#include <iostream>
#include <limits>
#include <cmath>
#include <cstdint>

namespace starml {
namespace models{
namespace classification{

classification::LogisticRegression::LogisticRegression(const starml::Matrix& train_data,
                                        const starml::Matrix& label,
                                        const float lambda) {
  std::cout << "LogisticRegression Model has been created with train_data" << "\n";
  this->param.lambda = lambda;
  this->optimizer.reset(new starml::optimizer::SGD());
  this->train(train_data, label);
}

classification::LogisticRegression::LogisticRegression(float lambda) {
  this->param.lambda = lambda;
  this->optimizer.reset(new starml::optimizer::SGD());
  std::cout << "LogisticRegression Model has been created with lambda = " << lambda << "\n";
}

classification::LogisticRegression::LogisticRegression(LogisticRegressionParam& param) {
  this->param = param;
  switch (param.solver_type) {
    case starml::optimizer::kSGD:
      this->optimizer.reset(new starml::optimizer::SGD(param.learning_rate));
      break;
    default:
      STARML_LOG(ERROR) << "Only support SGD optimizer now";
  }
  std::cout << "LogisticRegression Model has been created with param " << "\n";
}

classification::LogisticRegression::LogisticRegression(const starml::Matrix& train_data,
                                        const starml::Matrix& label,
                                        const starml::Matrix& weights,
                                        const float lambda) {
  this->optimizer.reset(new starml::optimizer::SGD());
  std::cout << "LogisticRegression Model has been created with weights" << "\n";
}

float classification::LogisticRegression::train(const starml::Matrix& train_data,
                                           const starml::Matrix& label) {
  std::cout << "Training LogisticRegression Model" << "\n";
  STARML_CHECK_EQ(train_data.dim(0), label.dim(0)) << "train_data and label should have same rows";
  // init parameters
  // need random module
  auto m = train_data.dim(0);
  auto n = train_data.dim(1);
  Matrix weight = full({n, 1}, train_data.device(), train_data.data_type(), 0.0);
  Matrix grad({n, 1}, train_data.device(), train_data.data_type());
  Matrix y_hat({m, 1}, label.device(), label.data_type());
  Matrix train_data_t = transpose(train_data);
  Matrix label_t = transpose(label);
  Matrix inverse_label_t = transpose(sub(1, label));
  Matrix loss_mat({1, 1}, train_data.device(), train_data.data_type());

  double diff = std::numeric_limits<double>::max();
  double current_loss = 0.0;
  double previous_loss = 0.0;
  int iter = 0;
  this -> optimizer -> set_param(weight, grad, param.learning_rate);
  //forward
  while ((iter < this -> param.max_iter) && (fabs(diff) > param.tolerance)) {
  //forward
  //backward -> grad
  // y^ = 1 / 1 + exp(x, w)
      div(float(1.0), add(float(1.0), exp(negtive(matmul(train_data, weight)))), y_hat);
  // loss = y * ln(y_hat) +  (1 - y) * ln(1 - y_hat)
      add(matmul(label_t, log(y_hat)), matmul(inverse_label_t, log(sub(1, y_hat))), loss_mat);
      current_loss = - loss_mat.data<float>()[0];
      diff = current_loss - previous_loss;
      previous_loss = current_loss;
  // grad = xt * (y^ - y) (n*m m*1)
      div(matmul(train_data_t, sub(y_hat, label)), m, grad);
      this -> optimizer -> step();
      iter += 1;
  }
  this -> parameters = weight;
  return 1;
}

float classification::LogisticRegression::train(const starml::Matrix& train_data,
                                           const starml::Matrix& label,
                                           const starml::Matrix& weights) {
  std::cout << "Training LogisticRegression Model with weights" << "\n";
  /*
  */
}

Matrix& classification::LogisticRegression::predict(const starml::Matrix& predict_data,
                                           starml::Matrix& predict_result) const {
  std::cout << "Predict LogisticRegression Model" << "\n";
  STARML_CHECK_EQ(predict_data.dim(1), this->parameters.dim(0)) << "The predict_data's features num should be the same as model weights ";

}

Matrix classification::LogisticRegression::predict(const starml::Matrix& predict_data) const {
  std::cout << "Predict LogisticRegression Model" << "\n";
  STARML_CHECK_EQ(predict_data.dim(1), this->parameters.dim(0)) << "The predict_data's features num should be the same as model weights ";
  Matrix y_hat = matmul(predict_data, this->parameters);
  Matrix predict_result = y_hat;
  return predict_result;
}

} // namespace classification
} // models
} // namespace starml
