#include <iostream>
#include "starml/basic/matrix.h"
#include "starml/operators/concat.h"
#include "starml/operators/transpose.h"
#include "gtest/gtest.h"

using namespace starml;

TEST(CONCAT, test){
  int m = 4;
  int n = 3;
  Matrix origin_data({m, n}, kCPU, kFloat);
  for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
          origin_data.data<float>()[i * n + j] = i + 1;
      }
  }
  origin_data.print();
  Matrix origin_data_concat = concat(origin_data, origin_data, 1);
  Matrix origin_data_concat1 = concat(origin_data, origin_data);
  origin_data_concat.print();
  origin_data_concat1.print();
  Matrix origin_data_cuda = origin_data.to(kCUDA);
  origin_data_cuda.print();
  Matrix origin_data_concat_cuda =  concat(origin_data_cuda, origin_data_cuda, 1);
  origin_data_concat_cuda.print();
  origin_data_cuda.print();
  Matrix origin_data_concat_cuda1 =  concat(origin_data_cuda, origin_data_cuda);
  origin_data_concat_cuda1.print();
  //test wrong
  // Matrix origin_data_concat_wrong = concat(origin_data, transpose(origin_data), 1);
  // Matrix origin_data_concat_wrong1 = concat(origin_data, transpose(origin_data), 1);
}
