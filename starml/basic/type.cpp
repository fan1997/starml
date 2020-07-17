#include "starml/basic/type.h"
#include "starml/utils/loguru.h"

namespace starml {
std::string to_string(DataTypeKind d, bool lower_case) {
  switch (d) {
    case DataTypeKind::Int:
      return lower_case ? "int" : "INT";
    case DataTypeKind::Float:
      return lower_case ? "float" : "FLOAT";
    case DataTypeKind::Double:
      return lower_case ? "double" : "DOUBLE";
    default:
      STARML_LOG(ERROR) << "Unknown data type: " << static_cast<int>(d);
      return "";
  }
}

std::ostream& operator<<(std::ostream& os, DataTypeKind type) {
  os << to_string(type, true);
  return os;
}

std::unordered_map<std::string, DataTypeKind> DataType::type_lists{
    {"int", kInt}, {"float", kFloat}, {"Double", kDouble}};
std::unordered_map<int, size_t> DataType::type_sizes{
    {0, sizeof(int)}, {1, sizeof(float)}, {2, sizeof(double)}};

DataType::DataType(DataTypeKind type) : type_(type) {}

size_t DataType::size() const {
  STARML_CHECK_NE(static_cast<int>(this->type_), -1)
      << "Data type is uncertain.";
  return type_sizes[static_cast<int>(this->type_)];
}
DataTypeKind DataType::type() const {
  STARML_LOG_IF(WARNING, (static_cast<int>(type_) == -1))
      << "Data type is uncertain.";
  return this->type_;
}

bool DataType::operator==(const DataType& rhs) const {
  return this->type_ == rhs.type_;
}
bool DataType::operator!=(const DataType& rhs) const {
  return !((*this) == rhs);
}

}  // namespace starml