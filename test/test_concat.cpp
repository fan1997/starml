#include <iostream>
#include "starml/basic/matrix.h"
#include "starml/operators/concat.h"
#include "gtest/gtest.h"

using namespace starml;

TEST(CONCAT, test){
  int m = 4;
  int n = 3;
  Matrix origin_data(m, n, kCPU, kFloat);
  for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
          origin_data.data<float>()[i * n + j] = i + 1;
      }
  }
  origin_data.print();
  Matrix origin_data_concat = concat(origin_data, origin_data, 1);
  origin_data_concat.print();
  Matrix origin_data_cuda = origin_data.to(kCUDA);
  Matrix origin_data_concat_cuda =  concat(origin_data_cuda, origin_data_cuda, 1);
  origin_data_concat_cuda.to(kCPU).print();

}
