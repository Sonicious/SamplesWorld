cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(CppMove VERSION 1.0 LANGUAGES CXX)

# use C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(HelloCppMove1 app1.cpp CppMove.cpp)
add_executable(HelloCppMove2 app2.cpp CppMove.cpp)
add_executable(HelloCppMove3 app3.cpp CppMove.cpp)