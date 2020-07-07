cmake_minimum_required(VERSION 2.8.12)

set(CMAKE_SUPPRESS_REGENERATION true)

project (lenet_mnist_image)

find_package(Caffe)

include_directories(${Caffe_INCLUDE_DIRS})

set(CAFFE_DIR "/home/yb/Desktop/caffe")
#list(APPEND CAFFE_DIR "/usr/include")
#list(APPEND CAFFE_DIR "/usr/local/include")
#list(APPEND CAFFE_DIR "/usr/local/cuda-10.2/include")
#list(APPEND CAFFE_DIR "/usr/local/include/opencv")
#list(APPEND CAFFE_DIR "/opt/OpenBLAS/include")
#list(APPEND CAFFE_DIR "/home/yb/Desktop/caffe/build_cmake/include")
include_directories(${CAFFE_DIR}/include)
include_directories(${CAFFE_DIR}/build)
include_directories(${CAFFE_DIR}/build/src)

add_definitions(${Caffe_DEFINITIONS})
add_executable(lenet_mnist_image lenet_mnist_image.cpp)
target_link_libraries(lenet_mnist_image ${Caffe_LIBRARIES})

