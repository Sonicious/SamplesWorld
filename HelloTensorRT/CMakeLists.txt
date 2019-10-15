cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(MinimalTensorRT VERSION 1.0 LANGUAGES CXX CUDA)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules/")

find_package(TensorRT REQUIRED 5.1)
find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(CUDART_LIBRARY nvinfer ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

message(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_executable( HelloTensorRT HelloTensorRT.cpp)
target_include_directories(HelloTensorRT PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(HelloTensorRT ${CUDART_LIBRARY} TensorRT::TensorRT)

add_executable( HelloCaffeImport HelloCaffeImport.cpp)
target_include_directories(HelloCaffeImport PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(HelloCaffeImport ${CUDART_LIBRARY} TensorRT::TensorRT)