#include <iostream>
#include <random>
#include "starml/basic/matrix.h"
#include "starml/models/linear_regression/linear_regression.h"
#include "starml/dataloader/dataloader.h"
#include "starml/operators/concat.h"
#include "gtest/gtest.h"
#include "starml/modelevaluator/metrics/metrics.h"

using namespace starml::modelevaluator::metrics;
using namespace starml::models::regression;
using namespace starml;
using std::default_random_engine;

TEST(LINEARREG, test){
#if 0
   LinearRegression model();

   int m = 100;
   int n = 30;
   starml::Matrix train_data({m, n}, kCPU, kFloat);
   starml::Matrix train_data_cat({m, 1}, kCPU, kFloat);
   starml::Matrix label({m, 1}, kCPU, kFloat);
   default_random_engine e;
   std::uniform_real_distribution<float> u(0, 1);
   for (size_t i = 0; i < m; i++) {
       label.mutable_data<float>()[i] = 0.0;
       train_data_cat.mutable_data<float>()[i] = 1.0;
       for (size_t j = 0; j < n; j++) {
           train_data.mutable_data<float>()[i * n + j] = u(e);
       }
       for (size_t j = 0; j < n; j++) {
           label.mutable_data<float>()[i] += (j + 1) * train_data.data<float>()[i * n + j]; // y = 1 * x1 + 2 * x2 + 3 * x3 ....
       }
   }
   train_data = concat(train_data, train_data_cat, 1);
   std::cout << "    ............. Training LR ............" << '\n' << '\n';
   std::cout << "    ..... Train_data ...." << '\n';
   train_data.print();
   std::cout << '\n';
   std::cout << "    ..... label ...." << '\n';
   label.print();
   std::cout << '\n';

// cpu

//train
   model.train(train_data, label);
//predict
   Matrix pred_label = model.predict(train_data);
//print
   std::cout << "    ..... pred_label ...." << '\n';
   pred_label.print();
   std::cout << '\n';
   Matrix weights = model.get_parameters();
   std::cout << "    .....lr model weights ...." << '\n';
   weights.print(); //[2,0]
   std::cout << '\n';

//evaluate
   metrics metric1;
   float err = metric1.mean_squared_error(label, pred_label);
   std::cout << "lr error: " << err << '\n';

//*******************************************************//
// GPU
   Matrix train_data_cuda = train_data.to(kCUDA);
   Matrix label_cuda = label.to(kCUDA);
// train
   model.train(train_data_cuda, label_cuda);
// predict
   Matrix pred_label_cuda = model.predict(train_data_cuda);
// print
   std::cout << "    ..... cuda pred_label ...." << '\n';
   pred_label_cuda.print();
   std::cout << '\n';
   weights = model.get_parameters();
   std::cout << "    .....cuda lr model weights ...." << '\n';
   weights.print(); //[2,0]
   std::cout << '\n';
// evaluate
   err = metric1.mean_squared_error(label_cuda, pred_label_cuda);
   std::cout << "cuda lr error: " << err << '\n';
#endif

//*******************************************************//
// real data
   DataLoader data_loader("/gpfs/share/home/1901213502/dataset/regression/housing_scale", kLIBSVM);
   // DataLoader data_loader;
   // data_loader.load_from_file("./example", kLIBSVM);
   long int total_size = data_loader.get_total_size();
   int num_instances = data_loader.get_num_instances();
   int num_features = data_loader.get_num_features();
   std::cout << "num_features: " << num_features << '\n';
   std::cout << "num_instances: " << num_instances << '\n';
   std::cout << "total_size: "<< total_size << '\n';

   Matrix label = data_loader.get_label();
   Matrix train_data = data_loader.get_data();
   Matrix train_data_cat({train_data.dim(0), 1}, kCPU, kFloat);
   for (size_t i = 0; i < train_data_cat.dim(0); i++) {
       train_data_cat.mutable_data<float>()[i] = 1;
   }
   train_data = concat(train_data, train_data_cat, 1);
   label.print();
   train_data.print();
   LinearRegression model;
 // cpu
 //train
   model.train(train_data, label);
 //predict
   Matrix pred_label = model.predict(train_data);
 //print
   std::cout << "    ..... pred_label ...." << '\n';
   pred_label.print();
   std::cout << '\n';
   Matrix weights = model.get_parameters();
   std::cout << "    .....lr model weights ...." << '\n';
   weights.print(); //[2,0]
   std::cout << '\n';

 //evaluate
   metrics metric1;
   float err = metric1.mean_squared_error(label, pred_label);
   std::cout << "lr error: " << err << '\n';
   float r2score = metric1.r2_score(label, pred_label);
   std::cout << "lr r2 score: " << r2score << '\n';

 //*******************************************************//
 // GPU
   Matrix train_data_cuda = train_data.to(kCUDA);
   Matrix label_cuda = label.to(kCUDA);
 // train
   model.train(train_data_cuda, label_cuda);
 // predict
   Matrix pred_label_cuda = model.predict(train_data_cuda);
 // print
   std::cout << "    ..... cuda pred_label ...." << '\n';
   pred_label_cuda.print();
   std::cout << '\n';
   weights = model.get_parameters();
   std::cout << "    .....cuda lr model weights ...." << '\n';
   weights.print(); //[2,0]
   std::cout << '\n';
 // evaluate
   err = metric1.mean_squared_error(label_cuda, pred_label_cuda);
   std::cout << "cuda lr error: " << err << '\n';
   r2score = metric1.r2_score(label_cuda, pred_label_cuda);
   std::cout << "cuda lr r2 score: " << r2score << '\n';
}
