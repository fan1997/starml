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
          origin_data.mutable_data<float>()[i * n + j] = 4;
          grad.mutable_data<float>()[i * n + j] = 1;
      }
  }
  origin_data.print();
  grad.print();
  std::shared_ptr<starml::optimizer::Optimizer> optimizer;
  optimizer.reset(new SGD());
  // SGD optimizer(origin_data, grad);
  // SGD optimizer;
  // optimizer.get_parameters().print();
  // optimizer.get_grad().print();
  // optimizer.set_param(origin_data, grad);
  //  optimizer.set_learning_rate(0.1);
  // optimizer.get_parameters().print();
  // optimizer.get_grad().print();
  // std::cout << optimizer.get_parameters().data<float>() << '\n';
  // std::cout << origin_data.data<float>() << '\n';

  optimizer -> set_param(origin_data, grad);
  optimizer -> set_learning_rate(0.001);
  // optimizer.step();
  optimizer -> step();
  // optimizer.get_parameters().print();
  // optimizer.get_grad().print();
  for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
          grad.mutable_data<float>()[i * n + j] = 2;
      }
  }
  optimizer -> step();
  optimizer -> step();
  optimizer -> get_parameters().print();
  optimizer -> get_grad().print();
  //
  // optimizer.step();
  // optimizer.step();
  // optimizer.get_parameters().print();
  // optimizer.get_grad().print();
  // origin_data.print();



  // SGD optimizer1(origin_data.to(kCUDA), grad.to(kCUDA), lr);
  // optimizer1.step();
  // optimizer1.get_parameters().print();
  // optimizer1.get_grad().print();
}
