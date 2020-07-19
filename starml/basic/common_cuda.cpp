#include "starml/basic/common_cuda.h"

namespace starml {

    const char* cublasGetErrorString(cublasStatus_t error) {
      switch (error) {
      case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";
      case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED";
      case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED";
      case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";
      case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH";
      case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR";
    #if CUDA_VERSION >= 6000
      case CUBLAS_STATUS_NOT_SUPPORTED:
        return "CUBLAS_STATUS_NOT_SUPPORTED";
    #if CUDA_VERSION >= 6050
      case CUBLAS_STATUS_LICENSE_ERROR:
        return "CUBLAS_STATUS_LICENSE_ERROR";
    #endif  // CUDA_VERSION >= 6050
    #endif  // CUDA_VERSION >= 6000
      }
      // To suppress compiler warning.
      return "Unrecognized cublas error string";
    }

    const char* cusparseGetErrorString(cusparseStatus_t error) {
      switch (error) {
        case CUSPARSE_STATUS_SUCCESS:
          return "CUSPARSE_STATUS_SUCCESS";
        case CUSPARSE_STATUS_NOT_INITIALIZED:
          return "CUSPARSE_STATUS_NOT_INITIALIZED";
        case CUSPARSE_STATUS_ALLOC_FAILED:
          return "CUSPARSE_STATUS_ALLOC_FAILED";
        case CUSPARSE_STATUS_INVALID_VALUE:
          return "CUSPARSE_STATUS_INVALID_VALUE";
        case CUSPARSE_STATUS_ARCH_MISMATCH:
          return "CUSPARSE_STATUS_ARCH_MISMATCH";
        case CUSPARSE_STATUS_MAPPING_ERROR:
          return "CUSPARSE_STATUS_MAPPING_ERROR";
        case CUSPARSE_STATUS_EXECUTION_FAILED:
          return "CUSPARSE_STATUS_EXECUTION_FAILED";
        case CUSPARSE_STATUS_INTERNAL_ERROR:
          return "CUSPARSE_STATUS_INTERNAL_ERROR";
        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
          return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
      }
      // To suppress compiler warning.
      return "Unrecognized cublas error string";
    }

    const char* cusolverGetErrorString(cusolverStatus_t error) {
      switch (error) {
      case CUSOLVER_STATUS_SUCCESS:
        return "CUSOLVER_STATUS_SUCCESS";
      case CUSOLVER_STATUS_NOT_INITIALIZED:
        return "CUSOLVER_STATUS_NOT_INITIALIZED";
      case CUSOLVER_STATUS_ALLOC_FAILED:
        return "CUSOLVER_STATUS_ALLOC_FAILED";
      case CUSOLVER_STATUS_INVALID_VALUE:
        return "CUSOLVER_STATUS_INVALID_VALUE";
      case CUSOLVER_STATUS_ARCH_MISMATCH:
        return "CUSOLVER_STATUS_ARCH_MISMATCH";
      case CUSOLVER_STATUS_EXECUTION_FAILED:
        return "CUSOLVER_STATUS_EXECUTION_FAILED";
      case CUSOLVER_STATUS_INTERNAL_ERROR:
        return "CUSOLVER_STATUS_INTERNAL_ERROR";
      case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
        return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
      }
      // To suppress compiler warning.
      return "Unrecognized cusolver error string";
    }

    const char* curandGetErrorString(curandStatus_t error) {
      switch (error) {
        case CURAND_STATUS_SUCCESS:
          return "CURAND_STATUS_SUCCESS";
        case CURAND_STATUS_VERSION_MISMATCH:
          return "CURAND_STATUS_VERSION_MISMATCH";
        case CURAND_STATUS_NOT_INITIALIZED:
          return "CURAND_STATUS_NOT_INITIALIZED";
        case CURAND_STATUS_ALLOCATION_FAILED:
          return "CURAND_STATUS_ALLOCATION_FAILED";
        case CURAND_STATUS_TYPE_ERROR:
          return "CURAND_STATUS_TYPE_ERROR";
        case CURAND_STATUS_OUT_OF_RANGE:
          return "CURAND_STATUS_OUT_OF_RANGE";
        case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
          return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
        case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
          return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
        case CURAND_STATUS_LAUNCH_FAILURE:
          return "CURAND_STATUS_LAUNCH_FAILURE";
        case CURAND_STATUS_PREEXISTING_FAILURE:
          return "CURAND_STATUS_PREEXISTING_FAILURE";
        case CURAND_STATUS_INITIALIZATION_FAILED:
          return "CURAND_STATUS_INITIALIZATION_FAILED";
        case CURAND_STATUS_ARCH_MISMATCH:
          return "CURAND_STATUS_ARCH_MISMATCH";
        case CURAND_STATUS_INTERNAL_ERROR:
          return "CURAND_STATUS_INTERNAL_ERROR";
      }
      // To suppress compiler warning.
      return "Unrecognized curand error string";
    }
}
