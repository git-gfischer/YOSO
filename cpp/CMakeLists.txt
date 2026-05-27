cmake_minimum_required(VERSION 3.18)
project(yoso_inference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

option(BUILD_TRT "Build TensorRT C++ inference binary" ON)
option(BUILD_ORT "Build ONNX Runtime C++ inference binary" ON)

find_package(OpenCV REQUIRED)

if(BUILD_TRT)
  add_subdirectory(trt)
endif()

if(BUILD_ORT)
  add_subdirectory(onnx)
endif()
