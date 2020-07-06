cmake_minimum_required(VERSION 2.8.8)

set(CAFFE_MAIN_LIB_DIR "/home/yb/Desktop/caffe/build/lib")
list(APPEND CMAKE_PREFIX_PATH ${CAFFE_MAIN_LIB_DIR})

set(CMAKE_SUPPRESS_REGENERATION true)

set(CAFFE_DIR "/home/yb/Desktop/caffe")
include_directories(${CAFFE_DIR}/build)
include_directories(${CAFFE_DIR}/build/include)
link_directories(${CAFFE_MAIN_LIB_DIR})

project (lenet_mnist_lmdb)

find_package(Caffe)

include_directories(${Caffe_INCLUDE_DIRS})
add_definitions(${Caffe_DEFINITIONS})

add_executable(lenet_mnist_lmdb lenet_mnist_lmdb.cpp)
target_link_libraries(lenet_mnist_lmdb ${Caffe_LIBRARIES})