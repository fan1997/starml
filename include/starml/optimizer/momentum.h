#pragma once
#include "starml/basic/matrix.h"
#include "starml/optimizer/optimizer.h"
#include "starml/operators/factories.h"

namespace starml {
namespace optimizer {

class Momentum : public Optimizer{
public:
    Momentum(float learning_rate = 0.001, float momentum = 0.9) : Optimizer(learning_rate), momentum(momentum){};
    Momentum(Matrix model_param, Matrix model_grad, float learning_rate = 0.001, float momentum = 0.9):
        Optimizer(model_param, model_grad, learning_rate), momentum(momentum) {
            accumulation = full(model_param.dims(),  model_param.device(), model_param.data_type(), 0.0);
        };
    void set_param(Matrix& model_param, Matrix& model_grad){
        std::cout << "momentum set param" << '\n';
        STARML_CHECK_DIMS_MATCH((model_param.dims() == model_grad.dims()));
        parameters = model_param;
        grad = model_grad;
        accumulation = full(model_param.dims(),  model_param.device(), model_param.data_type(), 0.0);
    };
    void set_momentum(float momentum = 0.9){this -> momentum = momentum;};
    float get_momentum() const {return this -> momentum;};
    float& get_momentum() {return this -> momentum;};
    Matrix& get_accumulation() {return this -> accumulation;}
    Matrix get_accumulation() const {return this -> accumulation;}
    void step();
private:
    float momentum;
    Matrix accumulation;
};

using momentum_op_kernel_fn = void (*)(Matrix& parameters, const Matrix& grad, Matrix& accumulation, const float lr, const float momentum);
STARML_DECLARE_DISPATCHER(momentum_dispatcher, momentum_op_kernel_fn);
void momentum_op(Matrix& parameters, const Matrix& grad, Matrix& accumulation, const float lr, const float momentum);

} // namespace optimizer
} // namespace starml
