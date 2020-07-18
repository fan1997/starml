#include <iostream>
#include "starml/basic/matrix.h"
#include "starml/operators/transpose.h"
#include "gtest/gtest.h"

using namespace starml;

TEST(TRANSPOSE, test){
  int m = 4;
  int n = 3;
  Matrix origin_data({m, n}, kCPU, kFloat);
  for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
          origin_data.data<float>()[i * n + j] = i + 1;
      }
  }
  origin_data.print();
  Matrix origin_data_trans = transpose(origin_data);
  origin_data_trans.print();
  Matrix origin_data_cuda = origin_data.to(kCUDA);
  Matrix origin_data_trans_cuda = transpose(origin_data_cuda);
  origin_data_trans_cuda.print();
}
