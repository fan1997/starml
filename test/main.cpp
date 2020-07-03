#include "gtest/gtest.h"
#include <iostream>
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  Matrix a(2, 3, kCPU);
  // float *data = a.data();
  // for (int i = 0; i < 2; i++) {
  //   for (int j = 0; j < 3; j++) {
  //     data[i * 3 + j] = i;
  //   }
  // }
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
  return RUN_ALL_TESTS();
}
