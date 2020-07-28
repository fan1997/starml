#include "starml/basic/handle_cuda.h"

namespace starml {
CUDAHandle::CUDAHandle(DeviceIndex index) {
  index_ = index;
  STARML_CUDA_CHECK(cudaStreamCreate(&stream_));
}
CUDAHandle::~CUDAHandle() { STARML_CUDA_CHECK(cudaStreamDestroy(stream_)); }
void CUDAHandle::synchronized() const {
  STARML_CUDA_CHECK(cudaStreamSynchronize(stream_));
}
void* CUDAHandle::stream() const { return stream_; }
void CUDAHandle::switch_device() const {
  STARML_CUDA_CHECK(cudaSetDevice(index_));
}
}  // namespace starml