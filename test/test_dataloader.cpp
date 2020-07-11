#include <iostream>
#include "starml/basic/matrix.h"
#include "starml/models/linear_regression/linear_regression.h"
#include "starml/dataloader/dataloader.h"
#include "gtest/gtest.h"


using namespace starml::regression;
using namespace starml;

TEST(ModelAndDataloader, test){

   LinearRegression model;
   LinearRegression model1(5.0);
   starml::Matrix train_data(2, 3, kCPU, kFloat);
   starml::Matrix label(2, 3, kCPU, kFloat);
   LinearRegression model2(train_data, label, 6.0);
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
   label_1.print();
   data_1.print();
}
