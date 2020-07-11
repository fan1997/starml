#include "starml/operators/factories.h"
namespace starml {
Matrix empty(int row, int col, DeviceType device_type, DataTypeKind data_type) {
  Matrix res(row, col, device_type, data_type);
  return res;
}
}  // namespace starml