#include <iostream>
#include "starml/basic/matrix.h"
#include "starml/modelevaluator/metrics/metrics.h"
#include "gtest/gtest.h"


using namespace starml::modelevaluator::metrics;
using namespace starml;

TEST(EVAL, test){
  metrics metric1;
  int label_size =  10;

  Matrix y({label_size, 1}, kCPU, kFloat);
  Matrix y_pred({label_size, 1}, kCPU, kFloat);
  for (int i = 0; i < label_size; i++) {
      y.data<float>()[i] = 1.0;
      y_pred.data<float>()[i] = 3.0;
  }
  float mse_err = metric1.mean_squared_error(y, y_pred);
  std::cout << "mse_err: " << mse_err << '\n';

  //cuda
  Matrix y_cuda = y.to(kCUDA);
  Matrix y_pred_cuda = y_pred.to(kCUDA);
  float mse_err_cuda = metric1.mean_squared_error(y_cuda, y_pred_cuda);
  std::cout << "mse_err_cuda: " << mse_err_cuda << '\n';


  for (int i = 0; i < label_size; i++) {
      y.data<float>()[i] = 1.0;
      y_pred.data<float>()[i] = i >= label_size / 2 ? 0.0 : 1.0;
  }
  float acc_score = metric1.accuracy_score(y, y_pred, false);
  std::cout << "acc_score: " << acc_score << '\n';
  // cuda
  acc_score = metric1.accuracy_score(y.to(kCUDA), y_pred.to(kCUDA));
  std::cout << "cuda_acc_score: " << acc_score << '\n';

}
