#include "gtest/gtest.h"
#include "starml/basic/matrix.h"
#include <iostream>
using namespace starml;

Matrix test() {
  Matrix a(3, 5, kCPU, kInt);
  int *data = a.data<int>();
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 5; j++) {
      data[i * 5 + j] = 1;
    }
  }
  return a;
}
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  Matrix a(2, 3, kCPU, kInt);
  int *data = a.data<int>();
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++) {
      data[i * 3 + j] = 3;
    }
  }
  Matrix b = test();
  a = b;
  int *data_a = a.data<int>();
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++) {
      std::cout << data_a[i * 3 + j] << " ";
    }
    std::cout << std::endl;
  } 
  // const int *dims = a.shape();
  // std::cout << dims[0] << " " << dims[1] << std::endl;
  return RUN_ALL_TESTS();
}