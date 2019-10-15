#include <iostream>
#include "mylib.hpp"

int main(int argc, char const *argv[])
{
  Mylib<float> a;
  a.data = 1.0f;
  Mylib<float> b(a);
  Mylib<double> c(a);
  Mylib<double> d(c);
  Mylib<double> e(3.0);
  Mylib<float> f(3.0f);
  return 0;
}