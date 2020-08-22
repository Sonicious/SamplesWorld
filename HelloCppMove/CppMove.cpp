#include "CppMove.hpp" 

#include <iostream>

// default constructor
CppMove::CppMove() noexcept
{
  std::cout << "default constructor" << std::endl;
  this->data = new int[10];
  for (int i = 0; i < 10; ++i)
  {
    this->data[i] = i;
  }
}

// copy constructor
CppMove::CppMove(const CppMove& other) noexcept
{
  std::cout << "Copy constructor" << std::endl;
  this->data = new int[10];
  for (int i = 0; i < 10; ++i)
  {
    this->data[i] = other.data[i];
  }
}

// move constructor
CppMove::CppMove(CppMove&& other) noexcept
{
  std::cout << "Move constructor" << std::endl;
  this->data = other.data;
  other.data = nullptr;
}

// copy assignment
CppMove& CppMove::operator=(const CppMove& other) noexcept
{
  std::cout << "copy assignment" << std::endl;
  if (this != &other)
  {
    if (this->data == nullptr)
    {
      this->data = new int[10];
    }
    for (int i = 0; i < 10; ++i)
    {
      this->data[i] = other.data[i];
    }
  }
  return *this;
}

// move assignment
CppMove& CppMove::operator=(CppMove&& other) noexcept
{
  std::cout << "move assignment" << std::endl;
  if (this != &other)
  {
    if(this->data != nullptr)
    {
      delete(this->data);
    }
    this->data = other.data;
    other.data = nullptr;
  }
  return *this;
}

// destructor
CppMove::~CppMove() noexcept
{
  std::cout << "default destructor" << std::endl;
  if (this->data)
  {
    delete(this->data);
  }
}