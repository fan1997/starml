#include "starml/optimizer/SGD.h"
#include <iostream>
namespace starml {
namespace optimizer {

STARML_DEFINE_DISPATCHER(sgd_dispatcher);
void sgd_op(Matrix& parameters, Matrix& grad, float lr) {
  sgd_dispatcher(parameters, grad, lr);
}

void SGD::step(){
  std::cout << "SGD step..." << '\n';
  // this->get_parameters().print();
  sgd_op(this->get_parameters(), this->get_grad(), this->get_learning_rate());
  // this->get_parameters().print();
}


} // namespace optimizer
} // namespace starml
