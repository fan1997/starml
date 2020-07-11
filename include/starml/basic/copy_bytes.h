#pragma once
#include <cstddef>
#include "starml/basic/device.h"
#include "starml/basic/macros.h"

namespace starml {
using CopyBytesFunction = void (*)(size_t nbytes, const void* src,
                                   Device src_device, void* dst,
                                   Device dst_device);
void copy_bytes(size_t nbytes, const void* src, Device src_device, void* dst,
                Device dst_device, bool async = false);

class RegisterCopyBytesFunction {
 public:
  RegisterCopyBytesFunction(DeviceType from_type, DeviceType to_type,
                            CopyBytesFunction func_sync,
                            CopyBytesFunction func_async);
};

#define STARML_REGISTER_COPY_BYTES_KERNEL(from, to, ...)                       \
  namespace {                                                                  \
  static RegisterCopyBytesFunction STARML_ANONYMOUS_VARIABLE(g_copy_function)( \
      from, to, __VA_ARGS__);                                                  \
  }
}