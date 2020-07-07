#pragma once
#include "starml/basic/type.h"
namespace starml {
#define STARML_PRIVATE_CASE_TYPE(enum_type, type, ...) \
  case enum_type: {                                    \
    using scalar_t = type;                             \
    return __VA_ARGS__();                              \
  }

#define STARML_DISPATCH_TYPES(SCALAR_TYPE, NAME, ...)                  \
  [&] {                                                                \
    switch (SCALAR_TYPE) {                                             \
      STARML_PRIVATE_CASE_TYPE(kInt, int, __VA_ARGS__)                 \
      STARML_PRIVATE_CASE_TYPE(kDouble, double, __VA_ARGS__)           \
      STARML_PRIVATE_CASE_TYPE(kFloat, float, __VA_ARGS__)             \
      default:                                                          \
        break;                                                        \
    }                                                                  \
  }()
}  // namespace starml