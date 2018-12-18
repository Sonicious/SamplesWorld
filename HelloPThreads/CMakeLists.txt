cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(Threading VERSION 1.0 LANGUAGES C)

find_package(Threads REQUIRED)

add_executable(HelloPThreads HelloPosixThreads.c)
target_link_libraries(HelloPThreads Threads::Threads)