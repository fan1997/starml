#include "starml/preprocessing/scaler/standardscaler_op.h"

namespace starml {
namespace preprocessing {
namespace scaler{
  STARML_DEFINE_DISPATCHER(stsfit_dispatcher);
  STARML_DEFINE_DISPATCHER(ststrans_dispatcher);
  STARML_DEFINE_DISPATCHER(stsinvtrans_dispatcher);

  void standardscaler_fit(const Matrix& origin_data, Matrix& mean_data,
                          Matrix& std_data) {
    stsfit_dispatcher(origin_data, mean_data, std_data);
  }

  void standardscaler_transform(const Matrix& origin_data, Matrix& result,
                          const Matrix& mean_data, const Matrix& std_data) {
    ststrans_dispatcher(origin_data, result, mean_data, std_data);
  }

  void standardscaler_inv_transform(const Matrix& transformed_data, Matrix& result,
                          const Matrix& mean_data, const Matrix& std_data) {
    stsinvtrans_dispatcher(transformed_data, result, mean_data, std_data);
  }

}
}
}
