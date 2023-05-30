
cmake_minimum_required(VERSION 3.22.1 FATAL_ERROR)
project(asynch_sw LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(simple_sw simple_sw.cpp)
target_link_libraries(simple_sw adept_lib_shared)
