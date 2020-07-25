#include <iostream>
#include "starml/basic/matrix.h"
#include "starml/operators/unary_ops.h"
#include "gtest/gtest.h"

using namespace starml;

TEST(UNARY, test){
  int m = 4;
  int n = 3;
  // Matrix origin_data({m, n}, kCPU, kFloat);
  Matrix origin_data({m, n}, kCPU, kInt32);
  for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
          origin_data.mutable_data<int>()[i * n + j] = 4;
      }
  }
  origin_data.print();

  Matrix res = exp(origin_data);
  res.print();
  // // Matrix origin_data_cuda = origin_data.to(kCUDA);
  // // Matrix res_cuda = exp(origin_data_cuda);
  // // res_cuda.print();
  //
  // Matrix res_sqrt = sqrt(origin_data);
  // res_sqrt.print();
  // // Matrix res_sqrt_cuda = sqrt(origin_data_cuda);
  // // res_sqrt_cuda.print();
  //
  // Matrix res_square = square(origin_data);
  // res_square.print();
  // // Matrix res_square_cuda = square(origin_data_cuda);
  // // res_square_cuda.print();
}
