

cmake_minimum_required(VERSION 3.3)

project(cmake-project-template)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 ")

set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})

set(DIVISIBLE_INSTALL_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(DIVISIBLE_INSTALL_BIN_DIR ${PROJECT_SOURCE_DIR}/bin)
set(DIVISIBLE_INSTALL_LIB_DIR ${PROJECT_SOURCE_DIR}/lib)

find_package(PkgConfig REQUIRED)

find_package(Armadillo REQUIRED)

pkg_check_modules(MLPACK REQUIRED mlpack)
include_directories(${MLPACK_INCLUDE_DIRS})
include_directories(${DIVISIBLE_INSTALL_INCLUDE_DIR})
include_directories(${DIVISION_HEADERS_DIR})



add_subdirectory(src)
# add_subdirectory(test)

