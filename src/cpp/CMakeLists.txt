cmake_minimum_required(VERSION 3.3)

project(cmake-project-template)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})

set(DIVISIBLE_INSTALL_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(DIVISIBLE_INSTALL_BIN_DIR ${PROJECT_SOURCE_DIR}/bin)
set(DIVISIBLE_INSTALL_LIB_DIR ${PROJECT_SOURCE_DIR}/lib)

find_package(PkgConfig REQUIRED)
find_package(Armadillo REQUIRED)
find_package(OpenCV REQUIRED)

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
set(LIBSVM_DIR /opt/homebrew/Cellar/libsvm/3.34/) 
include_directories(${LIBSVM_DIR}/include)
link_directories(${LIBSVM_DIR}/lib)

# Find and link XGBoost
find_library(XGBOOST_LIBRARY xgboost REQUIRED PATHS /opt/homebrew/lib /usr/local/lib /usr/lib)

find_library(ACCELERATE_FRAMEWORK Accelerate)
set(BLAS_LIBRARIES "/opt/homebrew/opt/openblas/lib/libopenblas.dylib")
set(LAPACK_LIBRARIES "/opt/homebrew/opt/openblas/lib/libopenblas.dylib")
pkg_check_modules(MLPACK REQUIRED mlpack)
pkg_check_modules(Armadillo REQUIRED Armadillo)
include_directories(${MLPACK_INCLUDE_DIRS})
include_directories(${DIVISIBLE_INSTALL_INCLUDE_DIR})
include_directories(${DIVISION_HEADERS_DIR})

add_subdirectory(src)
