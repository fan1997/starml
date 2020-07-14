#pragma once
#include <mutex>
#include <unordered_map>
#include <typeinfo>
#include "starml/basic/type.h"
#include "starml/basic/matrix.h"

namespace starml {

template <typename TFnPtr>
class Dispatcher;

template <typename TReturn, typename... TArgs>
class Dispatcher<TReturn(*)(TArgs...)> {
 public:
  using FnPtr = TReturn (*)(TArgs...);
  template <typename... TArgTypes>
  TReturn operator()(TArgTypes&&... args) {
    int key = dispatch_key(args...);
    FnPtr kernel = kernel_table_[key];
    return (*kernel)(std::forward<TArgTypes>(args)...);
  }
  void set_dispatcher(DeviceType device_type, FnPtr kernel) {
    std::lock_guard<std::mutex> guard(mu_);
    kernel_table_[static_cast<int>(device_type)] = kernel;
  }
 protected:
  template <typename T>
  int dispatch_key(const T& arg) {
    return static_cast<int>(arg.device_type().type());
  }

  template <typename THead, typename... TTail>
  int dispatch_key(const THead& head, const TTail&... tail) {
    if(typeid(head) == typeid(Matrix)) {
      return static_cast<int>(head.device_type().type());
    }
    return dispatch_key(tail...);
  }
  std::mutex mu_;
  std::unordered_map<int, FnPtr> kernel_table_;
};

template<typename Obj, typename FnPtr>
class DispatcherRegister {
 public:
  DispatcherRegister(DeviceType device_type, FnPtr kernel) {
    Obj::singleton().set_dispatcher(device_type, kernel);
  }
};

#define STARML_DECLARE_DISPATCHER(dispatcher, kernel_fn_type)  \
  class dispatcher##_t : public Dispatcher<kernel_fn_type> {   \
   public:                                                     \
    static dispatcher##_t& singleton() {                       \
      static dispatcher##_t dispatcher;                        \
      return dispatcher;                                       \
    }                                                          \
                                                               \
   private:                                                    \
    dispatcher##_t() {}                                        \
    dispatcher##_t(const dispatcher##_t&) = delete;            \
    dispatcher##_t& operator=(dispatcher##_t const&) = delete; \
  };                                                           \
  extern dispatcher##_t& dispatcher

#define STARML_DEFINE_DISPATCHER(dispatcher) \
  dispatcher##_t& dispatcher = dispatcher##_t::singleton()

#define STARML_REGISTER_KERNEL(dispatcher, device_type, fn) \
  static DispatcherRegister<dispatcher##_t, decltype(fn)> \
    register##dispatcher(device_type, fn)

#define STARML_PRIVATE_CASE_TYPE(enum_type, type, ...) \
  case enum_type: {                                    \
    using scalar_t = type;                             \
    return __VA_ARGS__();                              \
  }

#define STARML_DISPATCH_TYPES(SCALAR_TYPE, NAME, ...)        \
  [&] {                                                      \
    switch (SCALAR_TYPE) {                                   \
      STARML_PRIVATE_CASE_TYPE(kInt, int, __VA_ARGS__)       \
      STARML_PRIVATE_CASE_TYPE(kDouble, double, __VA_ARGS__) \
      STARML_PRIVATE_CASE_TYPE(kFloat, float, __VA_ARGS__)   \
      default:                                               \
        break;                                               \
    }                                                        \
  }()
}  // namespace starml