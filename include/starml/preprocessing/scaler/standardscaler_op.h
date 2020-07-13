#pragma once
#include "starml/basic/matrix.h"
#include "starml/basic/dispatch.h"

namespace starml {
namespace preprocessing {
namespace scaler{

using stsfit_kernel_fn = void (*)(const starml::Matrix& origin_data, starml::Matrix& mean_data,
                                  starml::Matrix& std_data);
STARML_DECLARE_DISPATCHER(stsfit_dispatcher, stsfit_kernel_fn);

void standardscaler_fit(const Matrix& origin_data, Matrix& mean_data,
                        Matrix& std_data);


using ststrans_kernel_fn = void (*)(const starml::Matrix& origin_data, starml::Matrix& result,
                                    const starml::Matrix& mean_data, const starml::Matrix& std_data);
STARML_DECLARE_DISPATCHER(ststrans_dispatcher, ststrans_kernel_fn);

void standardscaler_transform(const starml::Matrix& origin_data, starml::Matrix& result,
                        const starml::Matrix& mean_data, const starml::Matrix& std_data);


using stsinvtrans_kernel_fn = void (*)(const starml::Matrix& transformed_data, starml::Matrix& result,
                                    const starml::Matrix& mean_data, const starml::Matrix& std_data);
STARML_DECLARE_DISPATCHER(stsinvtrans_dispatcher, stsinvtrans_kernel_fn);

void standardscaler_inv_transform(const starml::Matrix& transformed_data, starml::Matrix& result,
                              const starml::Matrix& mean_data, const starml::Matrix& std_data);
} // namespace starml
} // namespace modelevaluator
} // namespace metrics
