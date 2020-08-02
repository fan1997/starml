#include <iostream>
#include "starml/basic/matrix.h"
#include "starml/operators/matmul.h"
#include "starml/operators/transpose.h"
#include "gtest/gtest.h"

using namespace starml;

TEST(MATMUL, test){
// mm
std::cout << "  MM:  " << '\n';
  int m = 3;
  int k = 2;
  int n = 2;
  Matrix origin_data({m, k}, kCPU, kFloat);
  for (int i = 0; i < m; i++) {
      for (int j = 0; j < k; j++) {
          origin_data.mutable_data<float>()[i * k + j] = i + 1;
      }
  }
  Matrix origin_data1({k, n}, kCPU, kFloat);
  for (int i = 0; i < k; i++) {
      for (int j = 0; j < n; j++) {
          origin_data1.mutable_data<float>()[i * n + j] = i;
      }
  }
  origin_data.print();
  origin_data1.print();
  Matrix origin_data1_trans = transpose(origin_data1);
  Matrix res = matmul(origin_data, origin_data1, kNoTrans, kNoTrans);
  res.print();
  //GPU
  Matrix origin_data_cuda = origin_data.to(kCUDA);
  Matrix origin_data1_cuda = origin_data1.to(kCUDA);
  Matrix origin_data1_cuda_trans = origin_data1_trans.to(kCUDA);
  Matrix res_cuda = matmul(origin_data_cuda, origin_data1_cuda, kNoTrans, kNoTrans);
  // Matrix res_cuda1 = matmul(origin_data_cuda, origin_data1_cuda_trans, kNoTrans, kTrans);
  res_cuda.print();
  // res_cuda1.print();

// dot
  std::cout << "  vv:  " << '\n';
  m = 1;
  k = 2;
  n = 1;
  Matrix vec1({m, k}, kCPU, kFloat);
  for (int i = 0; i < m; i++) {
      for (int j = 0; j < k; j++) {
          vec1.mutable_data<float>()[i * k + j] = i + 1;
      }
  }
  Matrix vec2({k, n}, kCPU, kFloat);
  for (int i = 0; i < k; i++) {
      for (int j = 0; j < n; j++) {
          vec2.mutable_data<float>()[i * n + j] = i;
      }
  }

  Matrix dot_res = matmul(vec1, vec2);
  dot_res.print();

  Matrix dot_res_cuda = matmul(vec1.to(kCUDA), vec2.to(kCUDA));
  dot_res_cuda.print();
//
  std::cout << "  mv:  " << '\n';
  m = 2;
  k = 2;
  n = 1;
  Matrix mat3({m, k}, kCPU, kFloat);
  for (int i = 0; i < m; i++) {
      for (int j = 0; j < k; j++) {
          mat3.mutable_data<float>()[i * k + j] = i + 1;
      }
  }
  Matrix vec3({k, n}, kCPU, kFloat);
  for (int i = 0; i < k; i++) {
      for (int j = 0; j < n; j++) {
          vec3.mutable_data<float>()[i * n + j] = i;
      }
  }

  Matrix mm_res = matmul(mat3, vec3);
  mm_res.print();
//
  Matrix mm_res_cuda = matmul(mat3.to(kCUDA), vec3.to(kCUDA));
  mm_res_cuda.print();
}
