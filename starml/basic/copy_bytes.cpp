#include "starml/basic/copy_bytes.h"
#include <iostream>
namespace starml {
static CopyBytesFunction g_copy_bytes[2][kNumDeviceTypes][kNumDeviceTypes];

RegisterCopyBytesFunction::RegisterCopyBytesFunction(
    DeviceType from_type, DeviceType to_type, CopyBytesFunction func_sync,
    CopyBytesFunction func_async) {
  auto from = static_cast<int>(from_type);
  auto to = static_cast<int>(to_type);
  g_copy_bytes[0][from][to] = func_sync;
  g_copy_bytes[1][from][to] = func_async;
}

void copy_bytes(size_t nbytes, const void* src, Device src_device, void* dst,
                Device dst_device, bool async) {
  auto ptr = g_copy_bytes[async ? 1 : 0][static_cast<int>(src_device.type())]
                         [static_cast<int>(dst_device.type())];
  ptr(nbytes, src, src_device, dst, dst_device);
}
}