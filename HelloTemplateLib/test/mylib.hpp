#pragma once

template<typename T>
class Mylib
{
public:
  Mylib();
  T data;

  template<typename I,
    typename std::enable_if_t<!std::is_same_v<I, T>> * = nullptr
  >
    Mylib(const Mylib<I>& other);

  Mylib(const Mylib<T>& other);

  template<typename I, typename T2 = T,
    typename std::enable_if<std::is_same<I, T2>::value, int>::type = 0
  >
    Mylib(const Mylib<I>& other);

  template<typename I,
    typename std::enable_if_t<std::is_same_v<I, T>> * = nullptr
  >
    Mylib(I i);
};

using FloatMylib = Mylib<float>;
using DoubleMylib = Mylib<double>;