#include "starml/preprocessing/scaler/standardscaler.h"
#include <iostream>

namespace starml {
namespace preprocessing {
namespace scaler{

void StandardScaler::fit(const starml::Matrix& origin_data){
  std::cout << "StandardScaler fitting..." << '\n';
  int features_num = origin_data.cols_num();
  Matrix mean_data = Matrix(1, features_num,
                  origin_data.device_type().type(), origin_data.data_type().type());
  Matrix std_data = Matrix(1, features_num,
                  origin_data.device_type().type(), origin_data.data_type().type());
  standardscaler_fit(origin_data, mean_data, std_data);
  this -> mean = mean_data;
  this -> std = std_data;
}

starml::Matrix StandardScaler::transform(const starml::Matrix& origin_data) const {
  std::cout << "StandardScaler transform..." << '\n';
  Matrix result = Matrix(origin_data.rows_num(), origin_data.cols_num(),
                  origin_data.device_type().type(), origin_data.data_type().type());
  standardscaler_transform(origin_data, result, this->mean, this->std);
  return result;
}

starml::Matrix StandardScaler::inverse_transform(const starml::Matrix& transformed_data) const {
  std::cout << "StandardScaler inverse_transform..." << '\n';
  Matrix result = Matrix(transformed_data.rows_num(), transformed_data.cols_num(),
                  transformed_data.device_type().type(), transformed_data.data_type().type());
  standardscaler_inv_transform(transformed_data, result, this->mean, this->std);
  return result;
}

} // namespace scaler
} // namespace preprocessing
} // namespace starml
