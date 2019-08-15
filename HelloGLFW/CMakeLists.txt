cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(HelloGLFW VERSION 1.0 LANGUAGES CXX)

# CMAKE Modules:
cmake_policy(SET CMP0072 NEW)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules/")
find_package(OpenGL)
find_package(GLFW3)

include_directories(${GLFW3_INCLUDE})
add_executable(HelloGLFW HelloGLFW.cpp)
target_link_libraries(HelloGLFW ${GLFW3_LIBRARY} OpenGL::GL )
