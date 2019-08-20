cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(HelloGLFW VERSION 1.0 LANGUAGES CXX)

# CMAKE Modules:
if(${CMAKE_VERSION} VERSION_LESS "3.11.0") 
    message("Please consider to switch to CMake 3.11.0 at least")
else()
    cmake_policy(SET CMP0072 NEW)
endif()
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules/")
find_package(OpenGL)
find_package(GLFW3)
include_directories(${GLFW3_INCLUDE})
add_subdirectory(glad/)

add_executable(HelloGLFW HelloGLFW.cpp)
target_link_libraries(HelloGLFW ${GLFW3_LIBRARY} OpenGL::GL glad )
