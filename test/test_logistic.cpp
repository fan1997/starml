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

//*******************************************************//
#if 1
// real data
   // DataLoader data_loader("/gpfs/share/home/1901213502/dataset/dataset-all/a9a", kLIBSVM);
   DataLoader data_loader("/home/amax101/hice/fanrb/star/dataset/classification/a9a", kLIBSVM);
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
   std::cout << "CPU time: " << timer.Elapsed() << '\n';
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
   std::cout << "CUDA time: " << gtimer.Elapsed() << '\n';
   std::cout << '\n';
 // evaluate
   metrics metric2;
   Matrix pred_label_cpu1 = pred_label_cuda.to(kCPU);
   for (size_t i = 0; i < pred_label_cpu1.size(); i++) {
     pred_label_cpu1.mutable_data<float>()[i] = pred_label_cpu1.mutable_data<float>()[i] >= 0 ? 1.0 : 0.0;
   }
   float cu_acc_score = metric2.accuracy_score(label, pred_label_cpu1);
   // std::cout << "CUDA lr acc_score: " << cu_acc_score << '\n';
   STARML_LOG(INFO) <<  "CUDA lr acc_score: " << cu_acc_score;
 #endif
#endif
}
