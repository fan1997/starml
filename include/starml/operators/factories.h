#pragma once
#include "starml/basic/matrix.h"

namespace starml {
  Matrix empty(int row, int col, DeviceType device_type, DataTypeKind data_type);
}