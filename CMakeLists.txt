# CMake makefile.
# To build the program, do
# mkdir -p build/
# cd build/
# cmake ..
# make

cmake_minimum_required (VERSION 2.6)
project (MDengine)

# Set the version number
set (VERSION_MAJOR 0)
set (VERSION_MINOR 4)

# Activate debug output of called with -DDEBUG=1
if (DEBUG)
    set (CMAKE_BUILD_TYPE "DEBUG")
    set (CUDA_VERBOSE_BUILD ON)
    set (CMAKE_VERBOSE_MAKEFILE ON)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -pedantic")
    set (CUDA_NVCC_DEBUG_FLAGS -Xcompiler -Wall;)
endif (DEBUG)

if (TEST)
    find_package(GTest REQUIRED)
    enable_testing()
endif (TEST)

if (CMAKE_BUILD_TYPE MATCHES "RELEASE")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif ()

set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fpic")

# Find CUDA
find_package(CUDA REQUIRED)
set (CUDA_PROPAGATE_HOST_FLAGS OFF)
set (custom_cuda_flags -std=c++11; -Xcompiler -fpic; --use_fast_math; --prec-div=true; --prec-sqrt=true;)
list (APPEND CUDA_NVCC_FLAGS ${custom_cuda_flags})
list (APPEND CUDA_NVCC_FLAGS ${CUDA_NVCC_DEBUG_FLAGS})
get_filename_component (CUDA_CUFFT_LIBRARY_PATH ${CUDA_CUFFT_LIBRARIES} DIRECTORY)

# Find Python libraries
find_package (PythonLibs 2.7 REQUIRED)
include_directories (${PYTHON_INCLUDE_DIRS})
get_filename_component (PYTHON_LIBRARY_PATH ${PYTHON_LIBRARIES} DIRECTORY)

# Find boost
find_package (Boost COMPONENTS python REQUIRED)

# Find Pugi XML
find_package (PugiXML REQUIRED)

message (STATUS "Building MD_engine Version ${VERSION_MAJOR}.${VERSION_MINOR}")


# Create configuration file
configure_file (
    "${PROJECT_SOURCE_DIR}/Config.h.in"
    "${PROJECT_BINARY_DIR}/Config.h"
)
include_directories ("${PROJECT_BINARY_DIR}")

set (MD_ENGINE_LIB_NAME Sim)

add_subdirectory (src)
#add src directory

# Install Python library
if (PYTHON)
    add_subdirectory (python)
    #add_custom_command (OUTPUT python/timestamp
    #    MAIN_DEPENDENCY python/Sim.cpp
    #    COMMAND ${PYTHON_BIN} python/setup.py build
    #    COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT})
endif (PYTHON)

# Add tests
if (TEST)
    add_subdirectory (unit_test)
endif (TEST)
