#include "gtest/gtest.h"
#include "starml/basic/matrix.h"
#include <iostream>
using namespace starml;

Matrix test() {
  Matrix a(3, 5, kCPU);
  float *data = a.data();
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 5; j++) {
      data[i * 5 + j] = 1;
    }
  }
  return a;
}
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  Matrix a(2, 3, kCPU);
  float *data = a.data();
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++) {
      data[i * 3 + j] = i;
    }
  }
  Matrix b = test();
  float *data_b = b.data();
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 5; j++) {
      std::cout << data_b[i * 5 + j] << " ";
    }
    std::cout << std::endl;
  } 
  return RUN_ALL_TESTS();
}