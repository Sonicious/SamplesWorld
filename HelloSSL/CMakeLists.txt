cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(MPI VERSION 1.0 LANGUAGES C)

find_package(OpenSSL REQUIRED)

add_executable(HelloMD5 HelloMD5.c)
target_include_directories(HelloMD5 PRIVATE ${OPENSSL_INCLUDE_DIR})
target_link_libraries(HelloMD5 ${OPENSSL_LIBRARIES})