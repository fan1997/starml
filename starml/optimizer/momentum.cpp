#include "starml/optimizer/momentum.h"
#include <iostream>
namespace starml {
namespace optimizer {

STARML_DEFINE_DISPATCHER(momentum_dispatcher);
void momentum_op(Matrix& parameters, const Matrix& grad, Matrix& accumulation, const float lr, const float momentum) {
  momentum_dispatcher(parameters, grad, accumulation, lr, momentum);
}

void Momentum::step(){
  STARML_LOG(INFO) << "Momentum step...";
  momentum_op(this->get_parameters(), this->get_grad(), this->get_accumulation(), this->get_learning_rate(), this->get_momentum());
}

} // namespace optimizer
} // namespace starml
