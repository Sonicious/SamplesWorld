#pragma once

class CppMove
{
public:
  // standard constructor
  CppMove();
  CppMove(const CppMove& other);
  CppMove(CppMove&& other);
  CppMove& operator=(const CppMove& other);
  CppMove& operator=(CppMove&& other);
  ~CppMove();
  int *data;
};
