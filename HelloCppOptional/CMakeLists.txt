cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(CppOptional VERSION 1.0 LANGUAGES CXX)

# use C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(HelloCppOptional app1.cpp)