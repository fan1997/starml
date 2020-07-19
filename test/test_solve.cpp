#include <iostream>
#include "starml/basic/matrix.h"
#include "starml/operators/solve.h"
#include "starml/operators/add_scalar.h"
#include "gtest/gtest.h"
#include <typeinfo>

using namespace starml;
TEST(SOLVE, test){
  int m = 2;
  int n = 2;
  Matrix A({m, n}, kCPU, kDouble);
  Matrix b({m, 1}, kCPU, kDouble);
  for (int i = 0; i < m; i++) {
      b.data<double>()[i] = 1;
      for (int j = 0; j < n; j++) {
          A.data<double>()[i * n + j] = j == i ? 1 : 0;
      }
  }
  A.print();
  Matrix solution_x = lu_solve(A, b);
  solution_x.print();
  // //gpu
  Matrix A_cuda = A.to(kCUDA);
  Matrix b_cuda = b.to(kCUDA);
  Matrix solution_x_cuda = lu_solve(A_cuda, b_cuda);
  solution_x_cuda.print();

  // Matrix B = add_scalar(A, 5);
  // B.print();
  // Matrix C =  add_scalar(A.to(kCUDA), 5);
  // C.print();

}
