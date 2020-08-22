#include "CppMove.hpp" 

#include <iostream>

// default constructor
CppMove::CppMove()
{
  std::cout << "default constructor" << std::endl;
  this->data = new int[10];
  for (int i = 0; i < 10; ++i)
  {
    this->data[i] = i;
  }
}

// copy constructor
CppMove::CppMove(const CppMove& other)
{
  std::cout << "Copy constructor" << std::endl;
  this->data = new int[10];
  for (int i = 0; i < 10; ++i)
  {
    this->data[i] = other.data[i];
  }
}

// move constructor
CppMove::CppMove(CppMove&& other)
{
  std::cout << "Move constructor" << std::endl;
  this->data = other.data;
  other.data = nullptr;
}

// copy assignment
CppMove& CppMove::operator=(const CppMove& other)
{
  std::cout << "copy assignment" << std::endl;
  if (this != &other)
  {
    if (!this->data)
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
CppMove& CppMove::operator=(CppMove&& other)
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
CppMove::~CppMove()
{
  std::cout << "default destructor" << std::endl;
  if (this->data)
  {
    delete(this->data);
  }
}