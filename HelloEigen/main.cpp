#include <iostream>
#include <complex>
#include <cstdlib>
#include <Eigen/Eigen>

int main(int argc, char const *argv[])
{
  Eigen::Matrix<float, 2, 2> a = Eigen::Matrix<float, 2, 2>::Zero();
  a(1,1) = 1;
  Eigen::Matrix<std::complex<double>, 2, 2> b, c;
  b << std::complex<double>(), std::complex<double>(1,1), std::complex<double>(2,2), std::complex<double>(3,3);
  c = a.cast<std::complex<double>>()*b;
  std::cout << "a = " << std::endl << a << std::endl;
  std::cout << "b = " << std::endl << b << std::endl;
  std::cout << "c = " << std::endl << c << std::endl;
  return EXIT_SUCCESS;
}
