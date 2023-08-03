cmake_minimum_required(VERSION 3.22.1 FATAL_ERROR)
project(adept LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set a default build type if none was specified
set(default_build_type "Release")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

option(BUILD_EXAMPLES "Building Examples" ON)
option(BUILD_TESTS "Building Tests" ON)

##########################################################################################
#       Setup STDEXEC
##########################################################################################

# this is a hack should be automatically detected from the CMAKE_PREFIX_PATH instead of manual
set(CPM_DOWNLOAD_VERSION 0.34.0)

if(CPM_SOURCE_CACHE)
  set(CPM_DOWNLOAD_LOCATION "${CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
elseif(DEFINED ENV{CPM_SOURCE_CACHE})
  set(CPM_DOWNLOAD_LOCATION "$ENV{CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
else()
  set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
endif()

if(NOT (EXISTS ${CPM_DOWNLOAD_LOCATION}))
  message(STATUS "Downloading CPM.cmake to ${CPM_DOWNLOAD_LOCATION}")
  file(DOWNLOAD
       https://github.com/TheLartians/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake
       ${CPM_DOWNLOAD_LOCATION}
  )
endif()

include(${CPM_DOWNLOAD_LOCATION})

CPMAddPackage(
  NAME stdexec
  GITHUB_REPOSITORY NVIDIA/stdexec
  GIT_TAG main # This will always pull the latest code from the `main` branch. You may also use a specific release version or tag
  OPTIONS
    "STDEXEC_ENABLE_CUDA OFF"
    "STDEXEC_BUILD_EXAMPLES OFF"
    "STDEXEC_BUILD_TESTS OFF"
    "STDEXEC_ENABLE_IO_URING_TESTS OFF"
    "BUILD_TESTING OFF"
)

if(stdexec_ADDED)
  link_directories(${stdexec_INSTALL_DIR}/lib64)
  include_directories(${stdexec_SOURCE_DIR}/include)
endif()

# need to add appropriate flags for stdexec
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --experimental-stdpar -std=c++20 --gcc-toolchain=/opt/cray/pe/gcc/12.2.0/bin/ -pthread")

##########################################################################################
#       Add subdirectories
##########################################################################################

# Add pyAdept
option(ADEPT_USE_PYTHON "Adding PyAdept" OFF)

if (ADEPT_USE_PYTHON)
    message(STATUS "Adding PyAdept support...")
    add_subdirectory(pyadept)
    message(STATUS "Adding external...")
    add_subdirectory(external)
endif()

include_directories("adept")

message(STATUS "Adding Adept library")

# Add adept library code
add_subdirectory(adept)

# Add examples
if(BUILD_EXAMPLES)
    message(STATUS "Adding Adept examples")
    add_subdirectory(examples)
endif()

# Add tests
if(BUILD_TESTS)
    message(STATUS "Adding ADEPT tests")
    include(CTest)

    # ADEPT tests

    add_test(simple_dna ./examples/simple_sw/simple_sw ../test-data/dna-reference.fasta ../test-data/dna-query.fasta ./simple_test_out ../test-data/expected256.algn)

    add_test(async_dna ./examples/asynch_sw/asynch_sw ../test-data/dna-reference.fasta ../test-data/dna-query.fasta ./simple_test_out ../test-data/expected256.algn)
    add_test(async_protein ./examples/asynch_protein/asynch_protein ../test-data/protein-reference.fasta ../test-data/protein-query.fasta ./simple_test_out ../test-data/protein_expected256.algn)

    add_test(multiGPU_dna ./examples/multi_gpu/multi_gpu ../test-data/dna-reference.fasta ../test-data/dna-query.fasta ./simple_test_out ../test-data/expected256.algn)

    add_test(multiGPU_protein ./examples/multigpu_protein/multigpu_protein ../test-data/protein-reference.fasta ../test-data/protein-query.fasta ./simple_test_out ../test-data/protein_expected256.algn)

    add_test(multiGPU_protein_scores ./examples/multigpu_protein_scoreonly/multigpu_protein_scoreonly ../test-data/protein-reference.fasta ../test-data/protein-query.fasta ./simple_test_out ../test-data/protein_scoreonly_expected256.algn)

    # PyADEPT testing

    if (ADEPT_USE_PYTHON)

        add_test(py_simple_dna python ./examples/py_examples/py_simple_sw -r ../test-data/dna-reference.fasta -q ../test-data/dna-query.fasta -t ../test-data/expected256.algn)

        add_test(py_async_dna python ./examples/py_examples/py_asynch_sw -r ../test-data/dna-reference.fasta -q ../test-data/dna-query.fasta -t ../test-data/expected256.algn)

        add_test(py_multiGPU_dna python ./examples/py_examples/py_multigpu_dna -r ../test-data/dna-reference.fasta -q ../test-data/dna-query.fasta -t ../test-data/expected256.algn)

        add_test(py_async_protein python ./examples/py_examples/py_asynch_protein -r ../test-data/protein-reference.fasta -q ../test-data/protein-query.fasta -t ../test-data/protein_expected256.algn)

        add_test(py_multiGPU_aa python ./examples/py_examples/py_multigpu_protein -r ../test-data/protein-reference.fasta -q ../test-data/protein-query.fasta -t ../test-data/protein_expected256.algn)

    endif()

endif()