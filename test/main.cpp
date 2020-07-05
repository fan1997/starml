#include <iostream>
#include <memory>
#include "gtest/gtest.h"
#include "starml/basic/matrix.h"
#include "starml/models/linear_regression/linear_regression.h"
#include "starml/dataloader/dataloader.h"

using namespace starml;
using namespace starml::regression;

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  const Matrix a(2, 3, kCPU, kInt);
  int *data = a.data<int>();
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++) {
      data[i * 3 + j] = 3.5;
    }
  }
  // Matrix b = test();
  // a = b;
  // int *data_a = a.data<int>();
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++) {
      std::cout << data[i * 3 + j] << " ";
    }
    std::cout << std::endl;
  }
  const int *dims = a.shape();
  std::cout << dims[0] << " " << dims[1] << std::endl;

  #if 0
   LinearRegression model;
   LinearRegression model1(5.0);
   starml::Matrix train_data(2, 3, kCPU);
   starml::Matrix label(2, 3, kCPU);
   LinearRegression model2(a, label, 6.0);
   model.train(train_data, label);
   std::cout << "model lambda: " << model.get_lambda() << '\n';
   std::cout << "model1 lambda: " << model1.get_lambda() << '\n';
   std::cout << "model2 lambda: " << model2.get_lambda() << '\n';


   // DataLoader data_loader("./example", kLIBSVM);

   DataLoader data_loader("/gpfs/share/home/1901213502/dataset/dataset-all/a1a", kLIBSVM);
   // DataLoader data_loader;
   // data_loader.load_from_file("./example", kLIBSVM);
   long int total_size = data_loader.get_total_size();
   int num_instances = data_loader.get_num_instances();
   int num_features = data_loader.get_num_features();
   std::cout << "num_features: " << num_features << '\n';
   std::cout << "num_instances: " << num_instances << '\n';
   std::cout << "total_size: "<< total_size << '\n';
   Matrix label_1 = data_loader.get_label();
   Matrix data_1 = data_loader.get_data();
   // return RUN_ALL_TESTS();
   for (int i = 0; i < num_instances; i++) {
     std::cout << label_1.data()[i] << ' ' << '\n';
   }
   for (size_t i = 0; i < num_instances; i++) {
     for (size_t j = 0; j < num_features; j++) {
        std::cout << data_1.data()[i * num_features + j] << ' ';
     }
     std::cout << '\n';
   }
 #endif
  return RUN_ALL_TESTS();
}
