#include <iostream>
#include <random>
#include "starml/basic/matrix.h"
#include "starml/models/logistic_regression/logistic_regression.h"
#include "starml/dataloader/dataloader.h"
#include "starml/operators/concat.h"
#include "gtest/gtest.h"
#include "starml/modelevaluator/metrics/metrics.h"
#include "starml/operators/binary_ops.h"
#include "starml/operators/unary_ops.h"
#include "starml/operators/factories.h"
#include "starml/utils/timer_cpu.h"
#include "starml/utils/timer_cuda.h"

using namespace starml::modelevaluator::metrics;
using namespace starml::models::classification;
using namespace starml;
using std::default_random_engine;

TEST(LOGISTIC, test){
#if 0
   LogisticRegressionParam lrparam;
   lrparam.learning_rate = 0.01;
   lrparam.solver_type = starml::optimizer::kSGD;
   lrparam.max_iter = 100;
   LogisticRegression model(lrparam);
   // LogisticRegression model;
   LogisticRegressionParam lrparam1 = model.get_param();
   std::cout << "model learning_rate: " << lrparam1.learning_rate << '\n';
   int m = 6;
   int n = 3;
   starml::Matrix train_data({m, n}, kCPU, kFloat);
   starml::Matrix train_data_cat({m, 1}, kCPU, kFloat);
   starml::Matrix label({m, 1}, kCPU, kFloat);
   default_random_engine e;
   std::uniform_real_distribution<float> u(0, 1);
   std::uniform_real_distribution<float> u1(-1, 0);
   for (size_t i = 0; i < m; i++) {
       label.mutable_data<float>()[i] = 0.0;
       train_data_cat.mutable_data<float>()[i] = 1.0;
       if(i < m/2){
           for (size_t j = 0; j < n; j++) {
               train_data.mutable_data<float>()[i * n + j] = u(e);
           }
       }else{
           for (size_t j = 0; j < n; j++) {
               train_data.mutable_data<float>()[i * n + j] = u1(e);
           }
       }

       for (size_t j = 0; j < n; j++) {
           label.mutable_data<float>()[i] += (j + 1) * train_data.data<float>()[i * n + j]; // y = 1 * x1 + 2 * x2 + 3 * x3 ....
       }
       label.mutable_data<float>()[i] = label.mutable_data<float>()[i] > 0.0 ? 1.0 : 0.0;
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
   // metrics metric1;
   // float err = metric1.(label, pred_label);
   // std::cout << "lr error: " << err << '\n';

//*******************************************************//
// // GPU
//    Matrix train_data_cuda = train_data.to(kCUDA);
//    Matrix label_cuda = label.to(kCUDA);
// // train
//    model.train(train_data_cuda, label_cuda);
// // predict
//    Matrix pred_label_cuda = model.predict(train_data_cuda);
// // print
//    std::cout << "    ..... cuda pred_label ...." << '\n';
//    pred_label_cuda.print();
//    std::cout << '\n';
//    weights = model.get_parameters();
//    std::cout << "    .....cuda lr model weights ...." << '\n';
//    weights.print(); //[2,0]
//    std::cout << '\n';
// // evaluate
//    err = metric1.mean_squared_error(label_cuda, pred_label_cuda);
//    std::cout << "cuda lr error: " << err << '\n';
#endif

//*******************************************************//
#if 1
// real data
   DataLoader data_loader("/gpfs/share/home/1901213502/dataset/dataset-all/a9a", kLIBSVM);
   // DataLoader data_loader("/gpfs/share/home/1901213502/dataset/dataset-all/w8a", kLIBSVM);
   // DataLoader data_loader("/gpfs/share/home/1901213502/dataset/dataset-all/covtype.libsvm.binary.scale", kLIBSVM);
   // DataLoader data_loader("/gpfs/share/home/1901213502/dataset/dataset-all/epsilon_normalized", kLIBSVM);
   // DataLoader data_loader("/gpfs/share/home/1901213502/dataset/dataset-all/epsilon_normalized_subset", kLIBSVM);

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
       label.mutable_data<float>()[i] = label.data<float>()[i] == -1.0 ? 0 : 1.0;
   }
   train_data = concat(train_data, train_data_cat, 1);
   label.print();
   train_data.print();

   LogisticRegressionParam lrparam;
   lrparam.learning_rate = 1.0;
   lrparam.solver_type = starml::optimizer::kSGD;
   lrparam.max_iter = 500;
   lrparam.tolerance = 0.0001;
   LogisticRegression model(lrparam);

 // cpu
 //train
   CpuTimer timer;
   timer.Start();
   model.train(train_data, label);
 //predict
   Matrix pred_label = model.predict(train_data);
   timer.Stop();
   std::cout << "TIME: " << timer.Elapsed() << '\n';
 //print
   std::cout << "    ..... pred_label ...." << '\n';
   pred_label.print();
   std::cout << '\n';
   Matrix weights = model.get_parameters();
   std::cout << "    .....lr model weights ...." << '\n';
   weights.print(); //[2,0]
   std::cout << '\n';

   Matrix pred_label_cpu = pred_label.to(kCPU);
   for (size_t i = 0; i < pred_label_cpu.size(); i++) {
       pred_label_cpu.mutable_data<float>()[i] = pred_label_cpu.mutable_data<float>()[i] >= 0 ? 1.0 : 0.0;
   }
   Matrix label_cpu = label.to(kCPU);
 //evaluate
   metrics metric1;
   float acc_score = metric1.accuracy_score(label_cpu, pred_label_cpu);
   std::cout << "lr acc_score: " << acc_score << '\n';

 //*******************************************************//
#if 1
 // GPU

   LogisticRegression model1(lrparam);
   Matrix train_data_cuda = train_data.to(kCUDA);
   Matrix label_cuda = label.to(kCUDA);
   label_cuda.print();
   GpuTimer gtimer;
   gtimer.Start();
 // train
   model1.train(train_data_cuda, label_cuda);
 // predict
   Matrix pred_label_cuda = model1.predict(train_data_cuda);
 // print
   gtimer.Stop();
   std::cout << "TIME: " << gtimer.Elapsed() << '\n';
   std::cout << '\n';
 // evaluate
   metrics metric2;
   Matrix pred_label_cpu1 = pred_label_cuda.to(kCPU);
   for (size_t i = 0; i < pred_label_cpu1.size(); i++) {
     pred_label_cpu1.mutable_data<float>()[i] = pred_label_cpu1.mutable_data<float>()[i] >= 0 ? 1.0 : 0.0;
   }
   float cu_acc_score = metric2.accuracy_score(label, pred_label_cpu1);
   std::cout << "cuda lr acc_score: " << cu_acc_score << '\n';
 #endif
#endif
}
