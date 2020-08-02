#include "starml/operators/matmul.h"
#include "starml/basic/common_cuda.h"
#include <iostream>
namespace starml {
namespace {

void vv(const Matrix& matrix1, const Matrix& matrix2,  Matrix& result){
    std::cout << "vv: " << '\n';
    //REPLACE WITH STAML_...
    cublasHandle_t handle;
    // cublasCreate(&handle);
    // cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
    STARML_CUBLAS_CHECK(cublasCreate(&handle));
    STARML_CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
    auto n = matrix1.dim(1);
    auto data_type = matrix1.data_type().type();
    const int incx = 1;
    const int incy = 1;
    switch (data_type) {
      case kFloat: {
        using scalar_t = float;
        const scalar_t *x = matrix1.data<scalar_t>();
        const scalar_t *y = matrix2.data<scalar_t>();
        scalar_t *r = result.mutable_data<scalar_t>();
        STARML_CUBLAS_CHECK(cublasSdot(handle, n, x, incx, y, incy, r));
        break;
      }
      case kDouble: {
        using scalar_t = double;
        const scalar_t *x = matrix1.data<scalar_t>();
        const scalar_t *y = matrix2.data<scalar_t>();
        scalar_t *r = result.mutable_data<scalar_t>();
        STARML_CUBLAS_CHECK(cublasDdot(handle, n, x, incx, y, incy, r));
        break;
      }
      default:
        STARML_LOG(ERROR) << "Unknown matmul type(only support float and double): " << static_cast<int>(data_type);
    }
}

void mv(const Matrix& matrix1, const Matrix& matrix2,  Matrix& result){
    std::cout << "mv: " << '\n';
    cublasHandle_t handle;
    // cublasCreate(&handle);
    // cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
    STARML_CUBLAS_CHECK(cublasCreate(&handle));
    STARML_CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    auto data_type = matrix1.data_type().type();
    const int m = matrix1.dim(1);
    const int n = matrix1.dim(0);
    const int lda = m;
    const int incx = 1;
    const int incy = 1;
    // transpose by default because of column priority
    switch (data_type) {
      case kFloat:{
        using scalar_t = float;
        const scalar_t *a = matrix1.data<scalar_t>();
        const scalar_t *x = matrix2.data<scalar_t>();
        scalar_t *y = result.mutable_data<scalar_t>();
        scalar_t alpha = 1, beta = 0;
        STARML_CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_T, m, n,
                    &alpha, a, lda, x, incx, &beta, y, incy));
        break;
      }
      case kDouble:{
        using scalar_t = double;
        const scalar_t *a = matrix1.data<scalar_t>();
        const scalar_t *x = matrix2.data<scalar_t>();
        scalar_t *y = result.mutable_data<scalar_t>();
        scalar_t alpha = 1, beta = 0;
        STARML_CUBLAS_CHECK(cublasDgemv(handle, CUBLAS_OP_T, m, n,
                    &alpha, a, lda, x, incx, &beta, y, incy));
        break;
      }
      default:
        STARML_LOG(ERROR) << "Unknown matmul type(only support float and double): " << static_cast<int>(data_type);
    }
}

void mm(const Matrix& matrix1, const Matrix& matrix2,  Matrix& result){
    std::cout << "mm: " << '\n';
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


void matmul_impl(const Matrix& matrix1, const Matrix& matrix2,  Matrix& result) {
    const int n_rows_mat1 = matrix1.dim(0);
    // const int n_cols_mat1 = matrix1.dim(1);
    // const int n_rows_mat2 = matrix2.dim(0);
    const int n_cols_mat2 = matrix2.dim(1);
    if((n_rows_mat1 == 1) && (n_cols_mat2 == 1)){
        vv(matrix1, matrix2, result);
    }else if((n_rows_mat1 != 1)&&(n_cols_mat2 == 1)){
        mv(matrix1, matrix2, result);
    }else{
        mm(matrix1, matrix2, result);
    }
}

}  // namespace
STARML_REGISTER_KERNEL(matmul_dispatcher, &matmul_impl, kCUDA, kCUDA, kCUDA);
}  // namespace starml
