#pragma once
#include "starml/basic/matrix.h"
#include "starml/preprocessing/scaler/standardscaler_op.h"

namespace starml {
namespace optimizer {

enum class SolverType: int16_t {
  SGD = 1,
  Momentum = 2,
  RmsProp = 3,
  Adam = 4
};
constexpr SolverType kSGD = SolverType::SGD;
constexpr SolverType kMomentum = SolverType::Momentum;
constexpr SolverType kRmsProp = SolverType::RmsProp;
constexpr SolverType kAdam = SolverType::Adam;

class Optimizer {
public:
    // Optimizer(){};
    Optimizer( float learning_rate = 0.001) : lr(learning_rate) {};
    Optimizer(Matrix model_param, Matrix model_grad, float learning_rate = 0.001):
              parameters(model_param), grad(model_grad), lr(learning_rate){
                  STARML_CHECK_DIMS_MATCH((model_param.dims() == model_grad.dims()));
              };

    void set_param(Matrix& model_param, Matrix& model_grad, float learning_rate = 0.001){
        parameters = model_param;
        grad = model_grad;
        lr = learning_rate;
    };
    void set_learning_rate(float learning_rate = 0.001){this -> lr = learning_rate;};
    Matrix& get_parameters()  {return parameters;};
    Matrix& get_grad()  {return grad;};
    float& get_learning_rate()  {return lr;};

    Matrix get_parameters() const {return parameters;};
    Matrix get_grad() const {return grad;};
    float get_learning_rate() const {return lr;};
    virtual void step() = 0;

protected:
    Matrix parameters;  // maybe should be a std::vector<Matrix> params;
    Matrix grad;
    float lr;
};

} // namespace optimizer
} // namespace starml
