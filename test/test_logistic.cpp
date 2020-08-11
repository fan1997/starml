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

// real data
//*******************************************************//
#if 1
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
   std::cout << "total_size: " << total_size << '\n';

   Matrix label = data_loader.get_label();
   Matrix train_data = data_loader.get_data();
   Matrix train_data_cat({train_data.dim(0), 1}, kCPU, kFloat);
   for (size_t i = 0; i < train_data_cat.dim(0); i++) {
       train_data_cat.mutable_data<float>()[i] = 1;
       label.mutable_data<float>()[i] = label.data<float>()[i] == -1.0 ? 0 : 1.0;
   }
   train_data = concat(train_data, train_data_cat, 1);
   train_data.print();
   label.print();
// cpu
#if 1
   LogisticRegressionParam lrparam;
   lrparam.learning_rate = 1.0;
   // lrparam.solver_type = starml::optimizer::kMomentum;
   lrparam.solver_type = starml::optimizer::kSGD;
   lrparam.max_iter = 500;
   lrparam.tolerance = 0.0001;
   LogisticRegression model(lrparam);

 //train
   CpuTimer timer;
   timer.Start();
   model.train(train_data, label);
 //predict
   Matrix pred_label = model.predict(train_data);
   timer.Stop();
   std::cout << "CPU TIME: " << timer.Elapsed() << "ms" << '\n';
 //evaluate
   metrics metric1;
   pred_label.print();
   float acc_score = metric1.accuracy_score(label, pred_label);
   std::cout << "lr acc_score: " << acc_score << '\n';
 #endif
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
   std::cout << "TIME: " << gtimer.Elapsed() << "ms" << '\n';

 // evaluate
   metrics metric2;
   pred_label_cuda.print();
   float cu_acc_score = metric2.accuracy_score(label_cuda, pred_label_cuda);
   std::cout << "cuda lr acc_score: " << cu_acc_score << '\n';
 #endif
#endif
}
