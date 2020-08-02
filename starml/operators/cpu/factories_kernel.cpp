#include "starml/operators/factories.h"
namespace starml {
namespace {
void full_impl(const Scalar& init_val, Matrix& result) {
  int size = result.size();
  auto dtype = result.data_type().type();
  STARML_DISPATCH_TYPES(dtype, "FULL_CPU", [&]() {
    auto data = result.mutable_data<scalar_t>();
    scalar_t value = init_val.value<scalar_t>();
    std::fill(data, data + size, value);
  });
}
}  // namespace
STARML_REGISTER_KERNEL(full_dispatcher, &full_impl, kCPU);
}  // namespace starml