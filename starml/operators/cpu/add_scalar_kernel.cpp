#include "starml/operators/add_scalar.h"

namespace starml {
namespace {
template <typename T>
void add_scalar_impl_kernel(T* data_1, T* res_data, double& b, int& size){
#pragma omp parallel for
  for (int i = 0; i < size; i++) {
    res_data[i] = data_1[i] + b;
  }
}
void add_scalar_impl(const Matrix& matrix1,  Matrix& result, double b) {
  // std::cout << "In add_impl " << std::endl;
  auto data_type = matrix1.data_type().type();
  int size = matrix1.size();
  STARML_DISPATCH_TYPES(data_type, "ADDSCALAR", [&]() {
    auto data_1 = matrix1.data<scalar_t>();
    auto res_data = result.data<scalar_t>();
    add_scalar_impl_kernel(data_1, res_data, b, size);
  });
}

}  // namespace

STARML_REGISTER_KERNEL(add_scalar_dispatcher, kCPU, &add_scalar_impl);

}  // namespace starml
