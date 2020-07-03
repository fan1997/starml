#include "starml/basic/device.h"

namespace starml {
Device::Device(DeviceType type) { this->type_ = type; }
DeviceType Device::type() { return this->type_; }
}  // namespace starml