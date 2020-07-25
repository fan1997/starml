#include "starml/models/logistic_regression/logistic_regression.h"
#include "starml/optimizer/optimizer.h"
#include "starml/optimizer/SGD.h"
#include "starml/operators/transpose.h"
#include "starml/operators/add_scalar.h"
#include "starml/operators/solve.h"
#include "starml/operators/matmul.h"
#include <iostream>

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
      this->optimizer.reset(new starml::optimizer::SGD());
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
  //init parameters
  // this->parameters = Matrix({train_data.dim(0), 1}, train_data.device(), train_data.data_type());
  // Matrix grad({train_data.dim(0), 1}, train_data.device(), train_data.data_type());

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

  // return predict_result;
}

Matrix classification::LogisticRegression::predict(const starml::Matrix& predict_data) const {
  std::cout << "Predict LogisticRegression Model" << "\n";
  STARML_CHECK_EQ(predict_data.dim(1), this->parameters.dim(0)) << "The predict_data's features num should be the same as model weights ";
  // Matrix predict_result = matmul(predict_data, this->parameters);
  // return predict_result;
}

} // namespace classification
} // models
} // namespace starml
