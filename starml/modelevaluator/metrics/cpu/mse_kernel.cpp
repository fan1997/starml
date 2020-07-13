#include "starml/modelevaluator/metrics/mse_op.h"
#include <omp.h>

namespace starml {
namespace modelevaluator {
namespace metrics{
namespace {
float mse_impl(const Matrix& y, const Matrix& y_pred) {
  auto data_type = y.data_type().type();
  int size = y.rows_num() * y.cols_num();
  float sum = 0.0;
  float score = 0.0;
  STARML_DISPATCH_TYPES(data_type, "MSE", [&]() {
    auto data1_ptr = y.data<scalar_t>();
    auto data2_ptr = y_pred.data<scalar_t>();
// #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < size; i++) {
      auto diff = data1_ptr[i] - data2_ptr[i];
      sum = sum + diff * diff;
    }
    score =  sum / size;
  });
  return score;
}
}  // namespace

STARML_REGISTER_KERNEL(mse_dispatcher, kCPU, &mse_impl);

}
}
}