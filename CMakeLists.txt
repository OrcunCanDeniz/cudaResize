cmake_minimum_required(VERSION 3.0.2)

add_compile_options(-std=c++14)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_MWAITXINTRIN_H_INCLUDED")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_FORCE_INLINES")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__STRICT_ANSI__")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

find_package(CUDA)

find_package(OpenCV REQUIRED)

include_directories(
  include
  ${OpenCV_LIBRARIES}
)

cuda_add_library(resize_cudaLib resize.cu)

add_executable(resize_test 
              main.cpp
  )

target_link_libraries(resize_test 
  ${OpenCV_LIBRARIES}
  resize_cudaLib
)
