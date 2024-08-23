cmake_minimum_required(VERSION 3.2)
project(crunner)

# add_subdirectory(division)
set(SOURCE_FILES cRunner.cpp)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(Armadillo REQUIRED)

add_executable(crunner ${SOURCE_FILES})
target_link_libraries(crunner ${XGBOOST_LIBRARY} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} ${ACCELERATE_FRAMEWORK})

install(TARGETS crunner DESTINATION ${DIVISIBLE_INSTALL_BIN_DIR})
