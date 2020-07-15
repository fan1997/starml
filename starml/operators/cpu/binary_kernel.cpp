#include "starml/operators/binary_ops.h"

namespace starml {
namespace {
void add_impl(const Matrix& matrix1, const Matrix& matrix2, Matrix& result) {
  // std::cout << "In add_impl " << std::endl;
  auto data_type = matrix1.data_type().type();
  int size = matrix1.size();
  STARML_DISPATCH_TYPES(data_type, "ADD", [&]() {
    auto data1 = matrix1.data<scalar_t>();
    auto data2 = matrix2.data<scalar_t>();
    auto res_data = result.data<scalar_t>();
    for (int i = 0; i < size; i++) {
      res_data[i] = data1[i] + data2[1];
    }
  });
}
}  // namespace

STARML_REGISTER_KERNEL(add_dispatcher, kCPU, &add_impl);

}  // namespace starml