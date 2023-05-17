# cmake min required by the stdexec lib
cmake_minimum_required(VERSION 3.22.1 FATAL_ERROR)
project(asynch_sw LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# this is a hack and this lib should be detected and set in the main CMakeLists.txt and not here
set(STDEXEC_PATH /global/homes/w/wwei/install/stdexec/)
set(CMAKE_CXX_FLAGS "--experimental-stdpar -std=c++20 --gcc-toolchain=/opt/cray/pe/gcc/12.2.0/bin/")

include_directories(${STDEXEC_PATH}/include)
link_directories(${STD_EXEC_PATH}/lib64)

# need to link to libpthread to use the stdexec thread pool scheduler
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

add_executable(asynch_sw asynch_sw.cpp)

# setting CXX_STANDARD again here
target_compile_features(asynch_sw INTERFACE cxx_std_20)
set_property(TARGET asynch_sw PROPERTY CXX_STANDARD 20)

target_link_libraries(asynch_sw adept_lib_shared)
