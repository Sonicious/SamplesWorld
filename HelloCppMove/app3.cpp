#include <iostream>
#include <utility>
#include <memory>

#include "CppMove.hpp"

CppMove myFunc(CppMove input)
{
  return input;
}

std::unique_ptr<CppMove> myGoodFunc(std::unique_ptr<CppMove> input)
{
  return input;
}

int main(int argc, char const *argv[])
{
  CppMove *A = new CppMove();
  CppMove *B = new CppMove(std::move(myFunc(std::move(*A))));
  delete(A);
  delete(B);
  std::cout << "raw pointer tests finished" << std::endl;

  std::unique_ptr<CppMove> S(new CppMove());
  std::unique_ptr<CppMove> T = myGoodFunc(std::move(S));
  return 0;
}