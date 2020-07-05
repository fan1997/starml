#include "starml/basic/type.h"
#include <iostream>

namespace starml {
std::unordered_map<std::string, DataTypeKind> DataType::type_lists{
    {"int", kInt}, {"float", kFloat}, {"Double", kDouble}};
std::unordered_map<std::string, size_t> DataType::type_sizes{
    {"int", sizeof(int)}, {"float", sizeof(float)}, {"Double", sizeof(double)}};

DataType::DataType(DataTypeKind type) : type_(type) {}
size_t DataType::size() const { return static_cast<size_t>(this->type_); }
bool DataType::operator==(const DataType &rhs) const {
  return this->type_ == rhs.type_;
}
}  // namespace starml