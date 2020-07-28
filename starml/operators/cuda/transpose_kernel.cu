#include "starml/operators/transpose.h"
#include "starml/basic/common_cuda.h"
#include <iostream>
namespace starml {
namespace {

void trans_impl(const Matrix& matrix1, Matrix& result) {
  auto data_type = matrix1.data_type().type();
  int rows_num = matrix1.dim(0);
  int cols_num = matrix1.dim(1);
  cublasHandle_t handle;
  STARML_CUBLAS_CHECK(cublasCreate(&handle));
  STARML_CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
  switch (data_type) {
      case kInt32:
        STARML_LOG(ERROR) << "(Temporary: )Unknown transpose type(only support float and double): " << static_cast<int>(data_type);
      case kFloat:{
        using scalar_t = float;
        const scalar_t *data1 = matrix1.data<scalar_t>();
        scalar_t *res_data = result.mutable_data<scalar_t>();
        scalar_t alpha = 1.;
        scalar_t beta  = 0.;
        STARML_CUBLAS_CHECK(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, rows_num, cols_num, &alpha, data1,
                    cols_num, &beta, data1, cols_num, res_data, rows_num));
        cudaDeviceSynchronize();
        break;
      }
      case kDouble:{
        using scalar_t = double;
        const scalar_t *data1 = matrix1.data<scalar_t>();
        scalar_t *res_data = result.mutable_data<scalar_t>();
        scalar_t alpha = 1.;
        scalar_t beta  = 0.;
        STARML_CUBLAS_CHECK(cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, rows_num, cols_num, &alpha, data1,
                    cols_num, &beta, data1, cols_num, res_data, rows_num));
        cudaDeviceSynchronize();
        break;
      }
      default:
        STARML_LOG(ERROR) << "Unknown transpose type(only support float and double): " << static_cast<int>(data_type);
  }

}
}  // namespace
STARML_REGISTER_KERNEL(transpose_dispatcher, &trans_impl, kCUDA, kCUDA);
}  // namespace starml
