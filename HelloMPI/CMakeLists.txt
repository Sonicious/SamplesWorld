cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(HelloMPI VERSION 1.0 LANGUAGES C)

find_package(MPI REQUIRED)

add_executable(HelloMPI HelloMPI.c )
target_include_directories(HelloMPI PRIVATE ${MPI_INCLUDE_PATH})
target_link_libraries(HelloMPI ${MPI_C_LIBRARIES})