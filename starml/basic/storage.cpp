#include "starml/basic/storage.h"
#include "starml/basic/device.h"

namespace starml {
  Storage::Storage(size_t size, DeviceType device_type, Allocator* allocator) {
    this->size_ = size;
    this->device_ = Device(device_type);
    this->allocator_ = allocator;
    this->data_ptr_ = allocator->allocate(size * sizeof(float));
  }

  Storage::~Storage() {
   
  }
}