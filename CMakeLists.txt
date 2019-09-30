CMAKE_MINIMUM_REQUIRED(VERSION 3.5)

project(WBLC)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC")
include_directories(${CMAKE_CURRENT_SOURCE_DIR})


#-------------------------------------------------------------------------------
# CUDA
#-------------------------------------------------------------------------------
find_package(CUDA 9.0 REQUIRED)
SET(CUDA_SEPARABLE_COMPILATION ON)

# compile
cuda_add_library(clusterCUDA src/cudaNeighbours/cudaNeighbours.cu)


#-------------------------------------------------------------------------------
# External Libraries
#-------------------------------------------------------------------------------
find_package(yaml-cpp REQUIRED)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

#-------------------------------------------------------------------------------
# Build Libraries
#-------------------------------------------------------------------------------
add_library(cluster	SHARED  src/cluster/WBLC.cpp)

target_link_libraries(cluster clusterCUDA
        ${CUDA_cusolver_LIBRARY} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cusolver_LIBRARIES})


#-------------------------------------------------------------------------------
# Build apps
#-------------------------------------------------------------------------------
add_executable(WBLCtesting	app/tester.cpp)
target_link_libraries(WBLCtesting cluster yaml-cpp)