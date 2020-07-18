#include <iostream>
#include "starml/basic/matrix.h"
#include "starml/preprocessing/scaler/standardscaler.h"
#include "gtest/gtest.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>

using namespace starml::preprocessing::scaler;
using namespace starml;

TEST(SCALER, test){
  StandardScaler scaler;
  int m = 3;
  int n = 3;
  Matrix origin_data({m, n}, kCPU, kFloat);
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
  Eigen::Matrix2d a;
  a << 1, 2, 3, 4;
  std::cout << "eigen a " << '\n';
  std::cout << a << '\n';

  std::vector<float> d0 = {2,3,4,5};
  std::vector<float> d1 = {1,2,};
  std::vector<float> result = {0,0};
  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> d_0(d0.data(), 2, 2);
  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
            d_1(d1.data(), 2, 1);
  std::cout << d_0 << '\n';
  std::cout << d_1 << '\n';
  // Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> retMat(result.data(), 2, 2);
  // retMat = d_0 * d_1;
  // std::cout << retMat << '\n';
  // for (size_t i = 0; i < 4; i++) {
  //     std::cout << result.data()[i] << '\n';
  // }
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> retMat = d_0 * d_1;
  std::cout << retMat << '\n';
  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >(result.data(),
                                                                                         retMat.rows(),
                                                                                         retMat.cols()) = retMat;
  for (size_t i = 0; i < 2; i++) {
    std::cout << result.data()[i] << '\n';
  }

}
