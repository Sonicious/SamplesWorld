#include "mylib_template.hpp"

template class Mylib<float>;
template Mylib<float>::Mylib(const Mylib<double>&);
template Mylib<float>::Mylib(float);
template class Mylib<double>;
template Mylib<double>::Mylib(const Mylib<float>&);
template Mylib<double>::Mylib(double);