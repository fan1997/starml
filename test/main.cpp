#include <memory>
#include "gtest/gtest.h"
#include "starml/basic/matrix.h"
#include "starml/basic/scalar.h"
#include "starml/operators/factories.h"
#include "starml/operators/binary_ops.h"
#include "starml/operators/unary_ops.h"
#include "starml/basic/handle_cuda.h"
using namespace starml;

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  // Matrix a = full({3, 1}, Device(kCPU), DataType(kInt32), 3);
  // Matrix b = full({3}, Device(kCPU), DataType(kFloat), 8.3);
  // Matrix result = empty({3, 1}, Device(kCUDA), DataType(kFloat));
  // exp(a.to(kCUDA), result);
  // result.print();
  // Matrix a_cuda = a.to(kCUDA);
  // Matrix b_cuda = b.to(kCUDA);
  // Matrix result_cuda = add(a_cuda, b_cuda);
  // result_cuda.print();
  // Matrix c = full({3, 1}, Device(kCPU), DataType(kFloat), 3.87);
  // c = exp(c);
  // c.print();
  // Matrix d = exp(cast(a, kFloat));
  // d.print();
  // int *data = a.mutable_data<int32_t>();
  // for (int i = 0; i < 1; i++) {
  //   for (int j = 0; j < 3; j++) {
  //     data[i * 3 + j] = 3;
  //   }
  // }
  // Matrix b = test();
  // Matrix a_cuda = a.to(kCUDA);
  // Matrix b_cuda = b.to(kCUDA);
  // Matrix result_cuda = add(a_cuda, b_cuda);
  // result_cuda.print();
  // result_cuda = add(10, b_cuda);
  // result_cuda.print();
  return RUN_ALL_TESTS();
}
