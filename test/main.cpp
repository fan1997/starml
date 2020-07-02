#include "gtest/gtest.h"
#include "starml/basic/Matrix.h"

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  starml::test();
  return RUN_ALL_TESTS();
}