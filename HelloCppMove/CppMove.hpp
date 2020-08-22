#pragma once

class CppMove
{
public:
  // standard constructor
  CppMove() noexcept;
  CppMove(const CppMove& other) noexcept;
  CppMove(CppMove&& other) noexcept;
  CppMove& operator=(const CppMove& other) noexcept;
  CppMove& operator=(CppMove&& other) noexcept;
  ~CppMove() noexcept;
  int *data;
};
