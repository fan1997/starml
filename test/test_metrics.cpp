#include <iostream>
#include "starml/basic/matrix.h"
#include "starml/modelevaluator/metrics/metrics.h"
#include "gtest/gtest.h"


using namespace starml::modelevaluator::metrics;
using namespace starml;

TEST(EVAL, test){
  metrics metric1;
  int label_size =  1000;
  Matrix y(label_size, 1, kCPU, kFloat);
  Matrix y_pred(label_size, 1, kCPU, kFloat);
  for (int i = 0; i < label_size; i++) {
      y.data<float>()[i] = 1.0;
      y_pred.data<float>()[i] = 3.0;
  }
  float score = metric1.mean_squared_error(y, y_pred);
  std::cout << "score: " << score << '\n';

  Matrix y_cuda = y.to(kCUDA);
  Matrix y_pred_cuda = y_pred.to(kCUDA);
  float score_cuda = metric1.mean_squared_error(y_cuda, y_pred_cuda);
  std::cout << "score_cuda: " << score_cuda << '\n';
}
