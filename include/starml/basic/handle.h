#pragma once
namespace starml {
class Handle {
 public:
  virtual void* stream() const = 0;
  virtual void synchronized() const = 0;
  virtual void switch_device() const = 0;
};
}  // namespace starml