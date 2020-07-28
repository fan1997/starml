#include "starml/operators/matmul.h"
#include "starml/basic/common_cuda.h"
#include <iostream>
namespace starml {
namespace {

void matmul_impl(const Matrix& matrix1, const Matrix& matrix2,  Matrix& result) {

  //REPLACE WITH STAML_...
  cublasHandle_t handle;
  // cublasCreate(&handle);
  // cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
  STARML_CUBLAS_CHECK(cublasCreate(&handle));
  STARML_CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
  // stream =
  //
  auto data_type = matrix1.data_type().type();
  const int n_rows_mat1 = matrix1.dim(0);
  const int n_cols_mat1 = matrix1.dim(1);
  const int n_rows_mat2 = matrix2.dim(0);
  const int n_cols_mat2 = matrix2.dim(1);
  const int m = n_cols_mat2;
  const int k = n_rows_mat2;
  const int n = n_rows_mat1;
  const int lda = n_cols_mat2;
  const int ldb = n_rows_mat2;
  const int ldc = n_cols_mat2;
  switch (data_type) {
      case kFloat:{
        using scalar_t = float;
        const scalar_t *data1 = matrix2.data<scalar_t>();
        const scalar_t *data2 = matrix1.data<scalar_t>();
        scalar_t *res_data = result.mutable_data<scalar_t>();
        scalar_t alpha = 1.;
        scalar_t beta  = 0.;
        STARML_CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, data1, lda, data2, ldb,
                    &beta, res_data, ldc));
        cudaDeviceSynchronize();
        break;
      }
      case kDouble:{
        using scalar_t = double;
        const scalar_t *data1 = matrix2.data<scalar_t>();
        const scalar_t *data2 = matrix1.data<scalar_t>();
        scalar_t *res_data = result.mutable_data<scalar_t>();
        scalar_t alpha = 1.;
        scalar_t beta  = 0.;
        STARML_CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, data1, lda, data2, ldb,
            &beta, res_data, ldc));
        cudaDeviceSynchronize();
        break;
      }
      default:
        STARML_LOG(ERROR) << "Unknown matmul type(only support float and double): " << static_cast<int>(data_type);
  }

}
}  // namespace
STARML_REGISTER_KERNEL(matmul_dispatcher, &matmul_impl, kCUDA, kCUDA);
}  // namespace starml
