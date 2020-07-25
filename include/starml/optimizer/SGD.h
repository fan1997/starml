#pragma once
#include "starml/basic/matrix.h"
#include "starml/optimizer/optimizer.h"

namespace starml {
namespace optimizer {

class SGD : public Optimizer{
public:
    SGD(){};
    SGD(Matrix model_param, Matrix model_grad, float learning_rate = 0.001):
        Optimizer(model_param, model_grad, learning_rate) {};
    void step();

private:
    Matrix parameters;  // maybe should be a std::vector<Matrix> params;
    Matrix grad;
    float lr;
};

using sgd_op_kernel_fn = void (*)(Matrix& parameters, Matrix& grad, float* lr);
STARML_DECLARE_DISPATCHER(sgd_dispatcher, sgd_op_kernel_fn);
void sgd_op(Matrix& parameters, Matrix& grad, float* lr);

} // namespace optimizer
} // namespace starml
