cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(HelloCL VERSION 1.0 LANGUAGES C)

find_package(OpenCL)
if(OpenCL_FOUND)
    add_executable(HelloCL HelloCL.c)
    target_link_libraries(HelloCL OpenCL::OpenCL)
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/kernel.cl DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endif()