#include <iostream>
#include "starml/basic/matrix.h"
#include "starml/models/linear_regression/linear_regression.h"
#include "starml/dataloader/dataloader.h"
#include "gtest/gtest.h"


using namespace starml::models::regression;
using namespace starml;

TEST(ModelAndDataloader, test){
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
