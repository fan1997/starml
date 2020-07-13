#pragma once
#include "starml/basic/matrix.h"
#include "starml/preprocessing/scaler/standardscaler_op.h"

namespace starml {
namespace preprocessing {
namespace scaler{

class StandardScaler {
public:
    StandardScaler(){};
    void fit(const starml::Matrix& origin_data); // need axis param
    starml::Matrix transform (const starml::Matrix& origin_data) const;
    starml::Matrix inverse_transform(const starml::Matrix& transformed_data) const;
    starml::Matrix get_mean() const { return this -> mean;}
    starml::Matrix get_std() const { return this -> std;}

private:
    starml::Matrix mean;
    starml::Matrix std;
};

} // namespace starml
} // namespace modelevaluator
} // namespace metrics
