#pragma once

#include <iostream>

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

  template<typename I,
    typename std::enable_if_t<std::is_same_v<I, T>> * = nullptr
  >
    Mylib(I i);
};

template<typename T>
Mylib<T>::Mylib()
{
  std::cout << "Standard" << std::endl;
}

template<typename T>
Mylib<T>::Mylib(const Mylib<T>& other)
{
  data = other.data;
  std::cout << "Same Type!" << std::endl;
}

template<typename T>
template<typename I,
  typename std::enable_if_t<std::is_same_v<I, T>> * = nullptr
>
Mylib<T>::Mylib(I i)
{
  std::cout << "anderer ctor" << std::endl;
  data = i;
}

template<typename T>
template<typename I,
  typename std::enable_if_t<!std::is_same_v<I, T>> * = nullptr
>
Mylib<T>::Mylib(const Mylib<I>& other)
{
  data = static_cast<T>(other.data);
  std::cout << "Different Type!" << std::endl;
}