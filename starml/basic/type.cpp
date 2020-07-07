#include "starml/basic/type.h"

namespace starml {
std::unordered_map<std::string, DataTypeKind> DataType::type_lists{
    {"int", kInt}, {"float", kFloat}, {"Double", kDouble}};
std::unordered_map<int, size_t> DataType::type_sizes{
    {0, sizeof(int)}, {1, sizeof(float)}, {2, sizeof(double)}};

DataType::DataType(DataTypeKind type) : type_(type) {}

size_t DataType::size() const {
  return type_sizes[static_cast<int>(this->type_)];
}
bool DataType::operator==(const DataType &rhs) const {
  return this->type_ == rhs.type_;
}

DataTypeKind DataType::type() const { return this->type_; }
}  // namespace starml