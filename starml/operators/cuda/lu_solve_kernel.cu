#include "starml/operators/solve.h"
#include "starml/basic/common_cuda.h"
#include <iostream>
namespace starml {
namespace {

void lu_solve_impl(const Matrix& matrix1, const Matrix& matrix2,  Matrix& result) {
  // matrix1 * result = matrix2;
  auto data_type = matrix1.data_type().type();
  const int n_rows_mat1 = matrix1.dim(0);
  const int n_cols_mat1 = matrix1.dim(1);
  const int n_rows_mat2 = matrix2.dim(0);
  const int n_cols_mat2 = matrix2.dim(1);
  STARML_CHECK_EQ(n_rows_mat1, n_cols_mat1) << "lu_solve matrix1 should be square";
  STARML_CHECK_EQ(n_rows_mat1, n_rows_mat2) << "lu_solve matrix1 and matrix2 should have same rows";
  STARML_CHECK_EQ(n_cols_mat2, 1) << "lu_solve matrix2 should have only one column";
  // need deep copy on cuda
  Matrix matrix1_t = matrix1.to(kCPU).to(kCUDA);
  result = matrix2.to(kCPU).to(kCUDA);
  Matrix matrix2_t = result;  // cause cusolver is inplace, directly change A & b
//cusolver
  cusolverDnHandle_t cusolverH = NULL;
  STARML_CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
  const int m = n_rows_mat1;
  const int lda = m;
  const int ldb = m;
// d_ipiv(P)(m*1) -> P*A = L*U
// d_info (1*1)
  Matrix dipiv = Matrix(matrix2.dims(), matrix2.device(), kInt32); // [m, 1], int
  Matrix dinfo = Matrix({1,1}, matrix2.device(), kInt32); // [1, 1], int
  int* d_Ipiv = dipiv.mutable_data<int>();
  int* d_info = dinfo.mutable_data<int>();
  int  lwork = 0;
  switch (data_type) {
      case kDouble:{
        using scalar_t = double;
        scalar_t *d_A = matrix1_t.mutable_data<scalar_t>();
        scalar_t *d_B = matrix2_t.mutable_data<scalar_t>();
        STARML_CUSOLVER_CHECK(cusolverDnDgetrf_bufferSize(cusolverH, m, m, d_A, lda, &lwork));
        Matrix dwork = Matrix({lwork,1}, matrix2.device(), kDouble); // [lwork , 1], int
        scalar_t* d_work = dwork.mutable_data<scalar_t>();
        STARML_CUSOLVER_CHECK(cusolverDnDgetrf(cusolverH, m, m, d_A, lda, d_work, d_Ipiv, d_info));
        STARML_CUSOLVER_CHECK(cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, m, 1, d_A, lda, d_Ipiv, d_B, ldb, d_info));
        break;
      }
      case kFloat:{
        using scalar_t = float;
        scalar_t *d_A = matrix1_t.mutable_data<scalar_t>();
        scalar_t *d_B = matrix2_t.mutable_data<scalar_t>();
        STARML_CUSOLVER_CHECK(cusolverDnSgetrf_bufferSize(cusolverH, m, m, d_A, lda, &lwork));
        Matrix dwork = Matrix({lwork,1}, matrix2.device(), kFloat); // [lwork , 1], int
        scalar_t* d_work = dwork.mutable_data<scalar_t>();
        STARML_CUSOLVER_CHECK(cusolverDnSgetrf(cusolverH, m, m, d_A, lda, d_work, d_Ipiv, d_info));
        STARML_CUSOLVER_CHECK(cusolverDnSgetrs(cusolverH, CUBLAS_OP_N, m, 1, d_A, lda, d_Ipiv, d_B, ldb, d_info));
        break;
      }
      default:
        STARML_LOG(ERROR) << "Unknown lu_solve type(only support float and double): " << static_cast<int>(data_type);
  }

}
}  // namespace
STARML_REGISTER_KERNEL(lu_solve_dispatcher, &lu_solve_impl, kCUDA, kCUDA);
}  // namespace starml
