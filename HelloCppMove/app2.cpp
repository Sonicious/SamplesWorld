#include <iostream>
#include <utility>
#include <memory>

#include "CppMove.hpp"

int main(int argc, char const *argv[])
{
  // raw pointer usage
  CppMove *A = new CppMove();
  CppMove *B = new CppMove();
  CppMove *C = new CppMove();
  *C = *A;
  *B = std::move(*A);
  delete(A);
  delete(B);
  delete(C);

  // Smart pointer usage (no copy here)
  std::unique_ptr<CppMove> S(new CppMove());
  S->data[0] = 42;
  std::unique_ptr<CppMove> T;
  T = std::move(S);
  std::cout << "value: " << T->data[0] << std::endl;
  
  return 0;
}