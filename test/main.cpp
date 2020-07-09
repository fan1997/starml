#include "gtest/gtest.h"
#include "starml/basic/matrix.h"
#include "starml/operators/binary_ops.h"
#include <iostream>
using namespace starml;

Matrix test() {
  Matrix a(2, 3, kCPU, kInt);
  int *data = a.data<int>();
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++) {
      data[i * 3 + j] = 1;
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
  Matrix result = add(a, b);
  result.print();
  return RUN_ALL_TESTS();
}