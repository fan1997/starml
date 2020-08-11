#include "starml/optimizer/SGD.h"
#include <iostream>
namespace starml {
namespace optimizer {

STARML_DEFINE_DISPATCHER(sgd_dispatcher);
void sgd_op(Matrix& parameters, const Matrix& grad, const float lr) {
  sgd_dispatcher(parameters, grad, lr);
}

void SGD::step(){
  STARML_LOG(INFO) << "SGD step...";
  sgd_op(this->get_parameters(), this->get_grad(), this->get_learning_rate());
}

} // namespace optimizer
} // namespace starml
