#pragma once

namespace starml {
enum class DeviceType : int { CPU = 0, CUDA = 1, NumDeviceTypes = 2};
constexpr DeviceType kCPU = DeviceType::CPU;
constexpr DeviceType kCUDA = DeviceType::CUDA;
constexpr int kNumDeviceTypes = static_cast<int>(DeviceType::NumDeviceTypes);

class Device {
 public:
  Device() {}
  Device(DeviceType type);
  DeviceType type();

 private:
  DeviceType type_;
};

}  // namespace starml