cmake_minimum_required(VERSION 2.8.8)

set(CAFFE_MAIN_LIB_DIR "/home/yb/Desktop/caffe/build/lib")
list(APPEND CMAKE_PREFIX_PATH ${CAFFE_MAIN_LIB_DIR})

set(CMAKE_SUPPRESS_REGENERATION true)

set(CAFFE_DIR "/home/yb/Desktop/caffe")
include_directories(${CAFFE_DIR}/build)
include_directories(${CAFFE_DIR}/build/include)
link_directories(${CAFFE_MAIN_LIB_DIR})

project (logreg)

find_package(Caffe)

include_directories(${Caffe_INCLUDE_DIRS})
include_directories(${Caffe_INCLUDE_DIRS}/layers)
add_definitions(${Caffe_DEFINITIONS})

add_executable(bindb bindb.cpp)
target_link_libraries(bindb ${Caffe_LIBRARIES})

add_executable(logreg_train logreg_train.cpp)
target_link_libraries(logreg_train ${Caffe_LIBRARIES})

add_executable(logreg_test logreg_test.cpp)
target_link_libraries(logreg_test ${Caffe_LIBRARIES})
