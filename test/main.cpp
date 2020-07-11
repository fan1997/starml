#include <memory>
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
  Matrix a_cuda = a.to(kCUDA);
  Matrix b_cuda = b.to(kCUDA);
  Matrix result_cuda = add(a_cuda, b_cuda);
  Matrix t = result_cuda.to(kCPU);
  t.print();
  return RUN_ALL_TESTS();
}
