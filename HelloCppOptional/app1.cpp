#include <iostream>
#include <optional>

int main(int argc, char const *argv[])
{
  std::optional<int> myNumber = 42;
  if (myNumber.has_value())
  {
    std::cout << "my Number is " << myNumber.value() << std::endl;
  }
  else
  {
    std::cout << "no number was saved in this optional" << std::endl;
  }
  myNumber.reset();
  if (myNumber.has_value())
  {
    std::cout << "my Number is " << myNumber.value() << std::endl;
  }
  else
  {
    std::cout << "no number was saved in this optional" << std::endl;
  }
  return 0;
}
