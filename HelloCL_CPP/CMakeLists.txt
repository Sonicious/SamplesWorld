cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(HelloCL VERSION 1.0 LANGUAGES CXX)

find_package(OpenCL)
if(OpenCL_FOUND)
    add_executable(HelloCL HelloCL.cpp)
    target_link_libraries(HelloCL OpenCL::OpenCL)
endif()