#include <iostream>
#include "starml/basic/matrix.h"
#include "starml/preprocessing/scaler/standardscaler.h"
#include "gtest/gtest.h"


using namespace starml::preprocessing::scaler;
using namespace starml;

TEST(SCALER, test){
  StandardScaler scaler;
  int m = 3;
  int n = 3;
  Matrix origin_data(m, n, kCPU, kFloat);
  for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
          origin_data.data<float>()[i * n + j] = i + 1;
      }
  }
  origin_data.print();
  scaler.fit(origin_data);
  Matrix mean_data = scaler.get_mean();
  Matrix std_data = scaler.get_std();
  mean_data.print();
  std_data.print();
  Matrix trans_data = scaler.transform(origin_data);
  trans_data.print();
  Matrix trans_trans_data = scaler.inverse_transform(trans_data);
  trans_trans_data.print();

  //GPU
  StandardScaler scaler1;
  Matrix origin_data_cuda = origin_data.to(kCUDA);
  origin_data_cuda.to(kCPU).print();
  scaler1.fit(origin_data_cuda);
  Matrix mean_data_cuda = scaler1.get_mean();
  Matrix std_data_cuda = scaler1.get_std();
  mean_data_cuda.to(kCPU).print();
  std_data_cuda.to(kCPU).print();
  Matrix trans_data_cuda = scaler1.transform(origin_data_cuda);
  trans_data_cuda.to(kCPU).print();
  Matrix trans_trans_data_cuda = scaler1.inverse_transform(trans_data_cuda);
  trans_trans_data_cuda.to(kCPU).print();

}
