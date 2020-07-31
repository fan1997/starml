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

  Matrix origin_data1 = log(res);
  origin_data1.print();

  origin_data1 = negtive(res);
  origin_data1.print();

  Matrix origin_data_cuda = origin_data.to(kCUDA);
  Matrix res_cuda = exp(origin_data_cuda);
  res_cuda.print();

  Matrix origin_data1_cuda = log(res_cuda);
  origin_data1_cuda.print();

  origin_data1_cuda = negtive(res_cuda);
  origin_data1_cuda.print();

}
