#pragma once

namespace starml {
enum class DeviceType : int { CPU = 0, CUDA = 1 };
constexpr DeviceType kCPU = DeviceType::CPU;
constexpr DeviceType kCUDA = DeviceType::CUDA;

class Device {
 public:
  Device() {}
  Device(DeviceType type);
  DeviceType type();

 private:
  DeviceType type_;
};

}  // namespace starml