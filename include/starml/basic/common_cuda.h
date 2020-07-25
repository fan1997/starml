#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cusparse.h>
#include <curand.h>
#include "starml/utils/loguru.h"

namespace starml {
#define STARML_CUDA_CHECK(condition) \
  cudaError_t error = condition;     \
  STARML_CHECK(error == cudaSuccess) << cudaGetErrorString(error);

#define STARML_CUBLAS_CHECK(condition)         \
do{                                            \
   cublasStatus_t status = condition;          \
   STARML_CHECK(status == CUBLAS_STATUS_SUCCESS) << cublasGetErrorString(status); \
} while(0)

#define STARML_CUSOLVER_CHECK(condition)         \
do{                                              \
   cusolverStatus_t status = condition;          \
   STARML_CHECK(status == CUSOLVER_STATUS_SUCCESS) << cusolverGetErrorString(status); \
}while(0)

const char* cublasGetErrorString(cublasStatus_t error);
const char* cusparseGetErrorString(cusparseStatus_t error);
const char* curandGetErrorString(curandStatus_t error);
const char* cusolverGetErrorString(cusolverStatus_t error);

#define STARML_CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

}  // namespace starml
