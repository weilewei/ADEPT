# cmake min required by the stdexec lib
cmake_minimum_required(VERSION 3.22.1 FATAL_ERROR)
project(asynch_sw LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(asynch_sw asynch_sw.cpp)

set_property(TARGET asynch_sw PROPERTY CXX_STANDARD 20)

target_link_libraries(asynch_sw adept_lib_shared)
