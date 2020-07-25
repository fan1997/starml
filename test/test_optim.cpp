#include <iostream>
#include "starml/basic/matrix.h"
#include "starml/optimizer/SGD.h"
#include "gtest/gtest.h"

using namespace starml;
using namespace starml::optimizer;
TEST(OPTIM, test){
  int m = 4;
  int n = 3;
  float lr = 0.001;
  Matrix origin_data({m, n}, kCPU, kFloat);
  Matrix grad({m, n}, kCPU, kFloat);
  for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
          origin_data.data<float>()[i * n + j] = 4;
          grad.data<float>()[i * n + j] = 1;
      }
  }
  origin_data.print();
  grad.print();
  SGD optimizer(origin_data, grad, lr);
  optimizer.step();
  optimizer.step();
  optimizer.step();
  optimizer.get_parameters().print();
  optimizer.get_grad().print();
  origin_data.print();

  // SGD optimizer1(origin_data.to(kCUDA), grad.to(kCUDA), lr);
  // optimizer1.step();
  // optimizer1.get_parameters().print();
  // optimizer1.get_grad().print();
}
