cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(HelloGraphics VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glm REQUIRED)
add_subdirectory(glad/)

#add_executable(HelloTextureDevicePointers HelloTextureDevicePointers.cu)
#add_executable(HelloTextureSurface HelloTextureSurface.cu)
#add_executable(HelloVerticeInteraction HelloVerticeInteraction.cu)
#add_executable(HelloPixelBufferInteraction HelloPixelBufferInteraction.cu)
#add_executable(HelloTexture HelloTexture.cpp)
add_executable(HelloModernGl HelloModernGl.cpp)

#target_link_libraries(HelloTextureDevicePointers glfw OpenGL::GL glad)
#target_link_libraries(HelloTextureSurface glfw OpenGL::GL glad)
#target_link_libraries(HelloVerticeInteraction glfw OpenGL::GL glad)
#target_link_libraries(HelloPixelBufferInteraction glfw OpenGL::GL glad)
#target_link_libraries(HelloTexture glfw OpenGL::GL glad)
target_link_libraries(HelloModernGl glfw OpenGL::GL glad glm::glm)