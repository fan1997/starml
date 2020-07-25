#include "starml/operators/matmul.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace starml {
namespace {
void matmul_impl(const Matrix& matrix1, const Matrix& matrix2, Matrix& result) {
  // std::cout << "In add_impl " << std::endl;
  auto data_type = matrix1.data_type().type();
  int m = matrix1.dim(0);
  int k = matrix1.dim(1);
  int n = matrix2.dim(1);
  STARML_DISPATCH_FLOATING_TYPES(data_type, "MATMUL", [&]() {
    auto data1 = matrix1.data<scalar_t>();
    auto data2 = matrix2.data<scalar_t>();
    auto res_data = result.mutable_data<scalar_t>();
    Eigen::Map<const Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mat1(data1, m, k);
    Eigen::Map<const Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
              mat2(data2, k, n);
    Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> resmat = mat1 * mat2;
    Eigen::Map<Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >(res_data,
                                                                                          resmat.rows(),
                                                                                          resmat.cols()) = resmat;
  });
}
}  // namespace

STARML_REGISTER_KERNEL(matmul_dispatcher, kCPU, &matmul_impl);

}  // namespace starml
