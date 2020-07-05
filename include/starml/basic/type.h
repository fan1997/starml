#pragma once
#include <cstddef>
#include <string>
#include <typeinfo>
#include <cxxabi.h>
#include <memory>
#include <unordered_map>

namespace starml {
enum class DataTypeKind : size_t { Int = 0, Float = 1, Double = 2 };

constexpr DataTypeKind kInt = DataTypeKind::Int;
constexpr DataTypeKind kFloat = DataTypeKind::Float;
constexpr DataTypeKind kDouble = DataTypeKind::Double;

template <typename T>
std::string type_name() {
  const char* name = typeid(T).name();
  int status = -1;
  std::unique_ptr<char, std::function<void(char*)>> demangled(
      abi::__cxa_demangle(name, nullptr, 0, &status), free);
  if (status == 0) {
    return demangled.get();
  } else {
    return name;
  }
}

class DataType {
 public:
  DataType() {}
  DataType(DataTypeKind type);
  size_t size() const;
  std::string type() const;
  template <typename T>
  bool is_valid() const {
    if (type_ == type_lists[type_name<T>()]) {
      return true;
    }
    return false;
  }
  bool operator==(const DataType &rhs) const;

 private:
  DataTypeKind type_;
  static std::unordered_map<std::string, DataTypeKind> type_lists;
  static std::unordered_map<std::string, size_t> type_sizes;
};

}  // namespace starml