#include <iostream>
#include "starml/basic/matrix.h"
#include "starml/operators/matmul.h"
#include "starml/operators/transpose.h"
#include "gtest/gtest.h"

using namespace starml;

TEST(MATMUL, test){
  int m = 3;
  int k = 2;
  int n = 1;
  Matrix origin_data(m, k, kCPU, kFloat);
  for (int i = 0; i < m; i++) {
      for (int j = 0; j < k; j++) {
          origin_data.data<float>()[i * k + j] = i + 1;
      }
  }
  origin_data.print();
  Matrix origin_data1(k, n, kCPU, kFloat);
  for (int i = 0; i < k; i++) {
      for (int j = 0; j < n; j++) {
          origin_data1.data<float>()[i * n + j] = i;
      }
  }
  origin_data.print();
  origin_data1.print();
  Matrix origin_data1_trans = transpose(origin_data1);
  Matrix res = matmul(origin_data, origin_data1, kNoTrans, kNoTrans);
  res.print();
  Matrix res1 = matmul(origin_data, origin_data1_trans, kNoTrans, kTrans);
  res1.print();

  //GPU
  Matrix origin_data_cuda = origin_data.to(kCUDA);
  Matrix origin_data1_cuda = origin_data1.to(kCUDA);
  Matrix origin_data1_cuda_trans = origin_data1_trans.to(kCUDA);
  Matrix res_cuda = matmul(origin_data_cuda, origin_data1_cuda, kNoTrans, kNoTrans);
  Matrix res_cuda1 = matmul(origin_data_cuda, origin_data1_cuda_trans, kNoTrans, kTrans);
  res_cuda.to(kCPU).print();
  res_cuda1.to(kCPU).print();

}
