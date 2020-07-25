#include "starml/basic/context_cuda.h"

namespace starml {
CUDAContext::CUDAContext(const Device& device) : gpu_idx_(device.index()) {}
CUDAContext::CUDAContext(int gpu_idx) : gpu_idx_(gpu_idx) {}
CUDAContext::~CUDAContext() {
  int n = streams_.size();
  for (int i = 0; i < n; i++) {
    STARML_CUDA_CHECK(cudaStreamDestroy(streams_[i]));
  }
}

void CUDAContext::synchronize() { STARML_CUDA_CHECK(cudaDeviceSynchronize()); }

void CUDAContext::set_counter(int count) { counter_ = count; }

cudaStream_t CUDAContext::stream() {
  if (counter_ >= streams_.size()) {
    cudaStream_t stream;
    STARML_CUDA_CHECK(cudaStreamCreate(&stream));
    streams_.push_back(stream);
    counter_++;
    return stream;
  }
  return streams_[counter_++];
}

void CUDAContext::prefetch_async(void* data, size_t size, cudaStream_t stream) {
  STARML_CUDA_CHECK(cudaMemPrefetchAsync(data, size, gpu_idx_, stream));
}

CUDAContextRegistry& CUDAContextRegistry::singleton() {
  static CUDAContextRegistry context_registry;
  return context_registry;
}

void CUDAContextRegistry::set_cuda_context(int gpu_idx,
                                           CUDAContext* cuda_context) {
  std::lock_guard<std::mutex> guard(mu_);
  cuda_contexts_[gpu_idx] = cuda_context;
}

CUDAContext* CUDAContextRegistry::cuda_context(int gpu_idx) {
  CUDAContext* context = cuda_contexts_[gpu_idx];
  STARML_CHECK_NOTNULL(context) << "Context for gpu " << gpu_idx << " is not set.";
  return context;
}

CUDAContextRegister::CUDAContextRegister() {
  int count;
  STARML_CUDA_CHECK(cudaGetDeviceCount(&count));
  for (int i = 0; i < count; i++) {
    CUDAContext *ctx = new CUDAContext(i);
    CUDAContextRegistry::singleton().set_cuda_context(i, ctx);
  }
}

static CUDAContextRegister g_cuda_context_;

CUDAContext* get_cuda_context(Device device) {
  int gpu_idx = device.index();
  STARML_CUDA_CHECK(cudaSetDevice(gpu_idx));
  CUDAContext* context = CUDAContextRegistry::singleton().cuda_context(gpu_idx);
  context->set_counter(0);
  return context;
}

}  // namespace starml