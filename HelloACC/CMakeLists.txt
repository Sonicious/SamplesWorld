cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(HelloACC VERSION 1.0 LANGUAGES C)

set(OpenACC_ACCEL_TARGET multicore)

find_package(OpenACC)
find_package(OpenMP)

if(OpenACC_C_FOUND AND OpenMP_C_FOUND)
  add_executable(HelloACC HelloACC.c)
  target_compile_options(HelloACC PUBLIC "${OpenACC_C_FLAGS}" "${OpenMP_C_FLAGS}")
  target_link_libraries(HelloACC PUBLIC OpenMP::OpenMP_C)
endif()