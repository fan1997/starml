#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "starml/basic/device.h"

namespace starml {
void copy_bytes_sync(size_t nbytes, const void* src, Device src_device,
                     void* dst, Device dst_device);
}
