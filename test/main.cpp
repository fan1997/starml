#include "gtest/gtest.h"
#include "starml/basic/Matrix.h"
#include "starml/models/linear_regression/linear_regression.h"
#include <iostream>

int main(int argc, char **argv) {
  // ::testing::InitGoogleTest(&argc, argv);
  // starml::test();
  // return RUN_ALL_TESTS();
  starml::test();
  using namespace starml::regression;
  LinearRegression model;
  LinearRegression model1(5.0);

  starml::Matrix train_data;
  starml::Matrix label;
  LinearRegression model2(train_data, label, 6.0);
  model.train(train_data, label);
  std::cout << "model lambda: " << model.get_lambda() << '\n';
  std::cout << "model1 lambda: " << model1.get_lambda() << '\n';
  std::cout << "model2 lambda: " << model2.get_lambda() << '\n';
}
