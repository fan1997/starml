#include <iostream>
#include "starml/basic/matrix.h"
#include "starml/models/linear_regression/linear_regression.h"
#include "starml/dataloader/dataloader.h"
#include "gtest/gtest.h"


using namespace starml::models::regression;
using namespace starml;

TEST(LINEARREG, test){
   LinearRegression model;
   int m = 3;
   int n = 2;
   starml::Matrix train_data({m, n}, kCPU, kFloat);
   starml::Matrix label({m, 1}, kCPU, kFloat);
   for (size_t i = 0; i < m; i++) {
       label.data<float>()[i] = 1.0;
       for (size_t j = 0; j < n; j++) {
           train_data.data<float>()[i * n + j] = i;
       }
   }
   model.train(train_data, label);
}
