cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(HelloGraphics VERSION 1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(glfw3 REQUIRED)
find_package(OpenGL 4.1 REQUIRED)
find_package(glm REQUIRED)
add_subdirectory(glad/)

add_custom_target(
  copy_shader_directory ALL
  COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_SOURCE_DIR}/shaders" "${CMAKE_BINARY_DIR}/shaders"
  COMMENT "Copy shader directory to build tree"
  VERBATIM
)

add_executable(HelloModernGl HelloModernGL.cpp)
add_executable(HelloVerticeInteraction HelloVerticeInteraction.cu)
add_executable(HelloTextureInteraction HelloTextureInteraction.cu)
add_executable(HelloTextureInteractionSurface HelloTextureInteractionSurface.cu)

target_link_libraries(HelloModernGl glfw OpenGL::GL glad glm::glm)
target_link_libraries(HelloVerticeInteraction glfw OpenGL::GL glad glm::glm)
target_link_libraries(HelloTextureInteraction glfw OpenGL::GL glad glm::glm)
target_link_libraries(HelloTextureInteractionSurface glfw OpenGL::GL glad glm::glm)

add_dependencies(HelloModernGl copy_shader_directory)
add_dependencies(HelloVerticeInteraction copy_shader_directory)
add_dependencies(HelloTextureInteraction copy_shader_directory)
add_dependencies(HelloTextureInteractionSurface copy_shader_directory)
