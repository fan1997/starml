#include "starml/operators/solve.h"
#include <omp.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace starml {
namespace {
void lu_solve_impl(const Matrix& matrix1, const Matrix& matrix2, Matrix& result) {
  auto data_type = matrix1.data_type().type();
  int m1_rows_num = matrix1.dim(0);
  int m1_cols_num = matrix1.dim(1);
  int m2_rows_num = matrix2.dim(0);
  int m2_cols_num = matrix2.dim(1);
  STARML_DISPATCH_FLOATING_TYPES(data_type, "LUSOLVE", [&]() {
    auto data_1 = matrix1.data<scalar_t>();
    auto data_2 = matrix2.data<scalar_t>();
    auto res_data = result.mutable_data<scalar_t>();
    // EIGEN (AX = b) A -1
    // check
    Eigen::Map<const Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> d_1(data_1, m1_rows_num, m1_cols_num);
    Eigen::Map<const Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> d_2(data_2, m2_rows_num, m2_cols_num);
    Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> solve_d =  d_1.partialPivLu().solve(d_2);
    Eigen::Map<Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >(res_data,
                                                                                           solve_d.rows(),
                                                                                           solve_d.cols()) = solve_d;
  });
}
}  // namespace

STARML_REGISTER_KERNEL(lu_solve_dispatcher, &lu_solve_impl, kCPU, kCPU);

}  // namespace starml
