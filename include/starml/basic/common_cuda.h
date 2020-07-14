#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "starml/basic/device.h"
#include "starml/utils/loguru.h"

namespace starml {
#define STARML_CUDA_CHECK(condition) \
  cudaError_t error = condition;     \
  STARML_CHECK(error == cudaSuccess) << cudaGetErrorString(error)

void copy_bytes_sync(size_t nbytes, const void* src, Device src_device,
                     void* dst, Device dst_device);
}  // namespace starml