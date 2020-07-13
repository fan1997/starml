#pragma once
#include <ostream>

namespace starml {
enum class DeviceType : int { CPU = 0, CUDA = 1, NumDeviceTypes = 2 };
constexpr DeviceType kCPU = DeviceType::CPU;
constexpr DeviceType kCUDA = DeviceType::CUDA;
constexpr int kNumDeviceTypes = static_cast<int>(DeviceType::NumDeviceTypes);

std::string to_string(DeviceType d, bool lower_case);
std::ostream& operator<<(std::ostream& stream, DeviceType type);

class Device {
 public:
  Device() {}
  Device(DeviceType type);
  DeviceType type();
  bool operator==(const Device& rhs);

 private:
  DeviceType type_;
};

}  // namespace starml