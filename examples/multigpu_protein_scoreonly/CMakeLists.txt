
cmake_minimum_required(VERSION 3.22.1 FATAL_ERROR)
project(asynch_protein_scores LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(multigpu_protein_scoreonly multigpu_protein_scoreonly.cpp)
target_link_libraries(multigpu_protein_scoreonly adept_lib_shared)
