#include "starml/basic/device.h"
#include "starml/utils/loguru.h"

namespace starml {
std::string to_string(DeviceType d, bool lower_case) {
  switch (d) {
    case DeviceType::CPU:
      return lower_case ? "cpu" : "CPU";
    case DeviceType::CUDA:
      return lower_case ? "cuda" : "CUDA";
    default:
      STARML_LOG(ERROR) << "Unknown device: " << static_cast<int>(d);
      return "";
  }
}

std::ostream& operator<<(std::ostream& os, DeviceType type) {
  os << to_string(type, true);
  return os;
}

Device::Device(DeviceType type) { this->type_ = type; }

DeviceType Device::type() { return this->type_; }

bool Device::operator==(const Device& rhs) {
  return this->type_ == rhs.type_;
}

}  // namespace starml