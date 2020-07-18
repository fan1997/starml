#include "starml/operators/transpose.h"
#include <omp.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace starml {
namespace {
void trans_impl(const Matrix& matrix1, Matrix& result) {
  // std::cout << "In trans_impl " << std::endl;
  auto data_type = matrix1.data_type().type();
  int rows_num = matrix1.dim(0);
  int cols_num = matrix1.dim(1);

  STARML_DISPATCH_TYPES(data_type, "TRANSPOSE", [&]() {
    auto data1 = matrix1.data<scalar_t>();
    auto res_data = result.data<scalar_t>();
    Eigen::Map<const Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> d1(data1, rows_num, cols_num);
    Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> trans_d = d1.transpose();
    Eigen::Map<Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >(res_data,
                                                                                           trans_d.rows(),
                                                                                           trans_d.cols()) = trans_d;
  });
}
}  // namespace

STARML_REGISTER_KERNEL(transpose_dispatcher, kCPU, &trans_impl);

}  // namespace starml
