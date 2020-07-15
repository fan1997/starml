#include "gtest/gtest.h"
#include "starml/basic/matrix.h"
#include "starml/operators/binary_ops.h"
#include "starml/basic/matrix_printer.h"
using namespace starml;

Matrix test() {
  Matrix a({2, 3}, Device(kCPU), DataType(kInt));
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
  Matrix a({2, 3}, Device(kCPU), DataType(kInt));
  int *data = a.data<int>();
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++) {
      data[i * 3 + j] = 3;
    }
  }
  MatrixPrinter mp;
  Matrix b = test();
  Matrix result = add(a, b);
  mp.print(result);
  Matrix a_cuda = a.to(kCUDA);
  Matrix b_cuda = b.to(kCUDA);
  Matrix result_cuda = add(a_cuda, b_cuda);
  Matrix t = result_cuda.to(kCPU);
  mp.print(t);
  return RUN_ALL_TESTS();
}