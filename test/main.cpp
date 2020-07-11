#include <iostream>
#include <memory>
#include "gtest/gtest.h"
#include "starml/basic/matrix.h"
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
  return RUN_ALL_TESTS();
}
