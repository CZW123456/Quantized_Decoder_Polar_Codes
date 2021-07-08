cmake_minimum_required(VERSION 3.5)
set(CMAKE_CXX_STANDARD 11)

set(PROJECT_NAME _libQuantizer)
project(${PROJECT_NAME})

add_subdirectory(pybind11)

include_directories(${PYBIND11_HEADERS})

find_package( OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})

if(UNIX AND NOT APPLE AND NOT ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang"))
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -fPIC -O3 -march=native -fopenmp")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk -std=c++11 -Wall -fPIC -O3 -march=native")
endif()

file(GLOB_RECURSE src_cxx "quantizers/_cpp/*.cpp")

pybind11_add_module(${PROJECT_NAME} SHARED ${src_cxx})

target_link_libraries(${PROJECT_NAME} PRIVATE pybind11::module ${OpenCV_LIBS})

set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}" SUFFIX "${PYTHON_MODULE_EXTENSION}")



