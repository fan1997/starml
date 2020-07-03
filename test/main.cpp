#include "gtest/gtest.h"
#include "starml/basic/matrix.h"
using namespace starml;

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  Matrix a(2, 3, kCPU);
  // float *data = a.data();
  // for (int i = 0; i < 2; i++) {
  //   for (int j = 0; j < 3; j++) {
  //     data[i * 3 + j] = i;
  //   }
  // }
  return RUN_ALL_TESTS();
}