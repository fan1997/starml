#include "starml/operators/factories.h"
namespace starml {
Matrix empty(Shape shape, Device device, DataType data_type) {
  Matrix res(shape, device, data_type);
  return res;
}
}  // namespace starml