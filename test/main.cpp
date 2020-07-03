#include "gtest/gtest.h"
#include "starml/basic/matrix.h"
#include "starml/models/linear_regression/linear_regression.h"

#include <iostream>
using namespace starml;
using namespace starml::regression;
int main(int argc, char **argv) {
  // ::testing::InitGoogleTest(&argc, argv);
  // starml::Matrix b;
  starml::Matrix a(2, 3, kCPU);
  float *data = a.data();
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++) {
      data[i * 3 + j] = i + 1 ;
    }
  }
  LinearRegression model;
  LinearRegression model1(5.0);

  starml::Matrix train_data(2, 3, kCPU);
  starml::Matrix label(2, 3, kCPU);
  LinearRegression model2(a, label, 6.0);
  model.train(train_data, label);
  std::cout << "model lambda: " << model.get_lambda() << '\n';
  std::cout << "model1 lambda: " << model1.get_lambda() << '\n';
  std::cout << "model2 lambda: " << model2.get_lambda() << '\n';
  // return RUN_ALL_TESTS();
}
