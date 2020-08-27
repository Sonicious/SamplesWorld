#include <iostream>
#include <utility>
#include <memory>

#include "CppMove.hpp"

int main(int argc, char const *argv[])
{
  // default ctor
  CppMove *A = new CppMove();
  // copy ctor
  CppMove *B = new CppMove(*A);
  // copy assign
  *A = *B;
  // move ctor
  CppMove *C = new CppMove(std::move(*A));
  // move assign
  *C = std::move(*B);

  // C has everything now, A and B are dead;
  if (A->data == nullptr)
  {
    std::cout << "A has no data" << std::endl;
  }
  delete(A);

  if (B->data == nullptr)
  {
    std::cout << "B has no data" << std::endl;
  }
  delete(B);

  if (C->data == nullptr)
  {
    std::cout << "C has no data" << std::endl;
  }
  delete(C);

  return 0;
}
