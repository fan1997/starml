#include "starml/modelevaluator/metrics/metrics_op.h"
#include <omp.h>

namespace starml {
namespace modelevaluator {
namespace metrics{
// mse
namespace {
template <typename T>
void mse_impl_kernel(const T* data1_ptr, const T* data2_ptr, int size, float& sum, float& score){
#pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < size; i++) {
      auto diff = data1_ptr[i] - data2_ptr[i];
      sum = sum + diff * diff;
    }
    score =  sum / size;
}
float mse_impl(const Matrix& y, const Matrix& y_pred) {
  auto data_type = y.data_type().type();
  int size = y.size();
  float sum = 0.0;
  float score = 0.0;
  STARML_DISPATCH_TYPES(data_type, "MSE", [&]() {
    auto data1_ptr = y.data<scalar_t>();
    auto data2_ptr = y_pred.data<scalar_t>();
    mse_impl_kernel(data1_ptr, data2_ptr, size, sum, score);
  });
  return score;
}
}  // namespace
STARML_REGISTER_KERNEL(mse_dispatcher, kCPU, &mse_impl);

// acc
namespace {
template <typename T>
void acc_impl_kernel(const T* data1_ptr, const T* data2_ptr, int size, float& sum){
#pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < size; i++) {
      sum += (data1_ptr[i] == data2_ptr[i]);
    }
}
float acc_impl(const Matrix& y, const Matrix& y_pred) {
  auto data_type = y.data_type().type();
  int size = y.size();
  float sum = 0.0;
  STARML_DISPATCH_TYPES(data_type, "MSE", [&]() {
    auto data1_ptr = y.data<scalar_t>();
    auto data2_ptr = y_pred.data<scalar_t>();
    acc_impl_kernel(data1_ptr, data2_ptr, size, sum);
  });
  return sum;
}
}  // namespace
STARML_REGISTER_KERNEL(acc_dispatcher, kCPU, &acc_impl);


}
}
}
