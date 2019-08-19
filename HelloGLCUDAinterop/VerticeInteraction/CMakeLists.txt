cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(HelloGlCudaInterop VERSION 1.0 LANGUAGES CXX CUDA)

# CMAKE Modules:
if(${CMAKE_VERSION} VERSION_LESS "3.11.0")
    message("Please consider to switch to CMake 3.11.0")
else()
    cmake_policy(SET CMP0072 NEW) # to get modern OpenGL
endif()
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules/")

find_package(OpenGL)
find_package(GLFW3)

include_directories(${GLFW3_INCLUDE})
add_executable(HelloGlCudaInterop HelloGlCudaInterop.cu)
target_link_libraries(HelloGlCudaInterop ${GLFW3_LIBRARY} OpenGL::GL )
