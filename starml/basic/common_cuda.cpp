#include "starml/basic/common_cuda.h"
#include "starml/basic/copy_bytes.h"

namespace starml {
// void copy_bytes_async(size_t nbytes, const void* src, Device src_device,
//                       void* dst, Device dst_device) {
//   cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyDefault);
// }

void copy_bytes_sync(size_t nbytes, const void* src, Device src_device,
                     void* dst, Device dst_device) {
  cudaMemcpy(dst, src, nbytes, cudaMemcpyDefault);
}

STARML_REGISTER_COPY_BYTES_KERNEL(DeviceType::CPU, DeviceType::CUDA,
                                  copy_bytes_sync, nullptr);
STARML_REGISTER_COPY_BYTES_KERNEL(DeviceType::CUDA, DeviceType::CPU,
                                  copy_bytes_sync, nullptr);
}