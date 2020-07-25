#pragma once
#include <vector>
#include <unordered_map>
#include <mutex>
#include "starml/basic/common_cuda.h"
#include "starml/basic/device.h"

namespace starml {
class CUDAContext {
 public:
  explicit CUDAContext(const Device& device);
  explicit CUDAContext(int gpu_idx_);
  ~CUDAContext();
  void synchronize();
  cudaStream_t stream();
  void set_counter(int count);
  void prefetch_async(void* data, size_t size, cudaStream_t stream);
  DeviceIndex gpu_idx() { return gpu_idx_; }

 private:
  DeviceIndex gpu_idx_;
  std::vector<cudaStream_t> streams_;
  int counter_;
};

class CUDAContextRegistry {
 public:
  static CUDAContextRegistry& singleton(); 
  void set_cuda_context(int gpu_idx, CUDAContext* cuda_context);
  CUDAContext* cuda_context(int gpu_idx);
 private:
  CUDAContextRegistry() = default;
  CUDAContextRegistry(const CUDAContextRegistry&) = delete;
  CUDAContextRegistry& operator=(const CUDAContextRegistry&) = delete;
  std::unordered_map<int, CUDAContext*> cuda_contexts_;
  std::mutex mu_;
};

class CUDAContextRegister{
 public:
  CUDAContextRegister();
};

CUDAContext* get_cuda_context(Device device);

}  // namespace starml