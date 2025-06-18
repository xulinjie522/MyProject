cmake_minimum_required(VERSION 3.14)
project(multi_model LANGUAGES CXX CUDA)

# Global options
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

# CUDA Compiler
find_package(CUDA REQUIRED)

# OpenCV
find_package(OpenCV REQUIRED)

# nlohmann_json
find_package(nlohmann_json 3.2.0 REQUIRED)

#udp
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/udp/include)

# TensorRT: 可根据平台修改
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    message(STATUS "AARCH64 detected")
    include_directories(/usr/local/cuda/targets/aarch64-linux/include)
    link_directories(/usr/local/cuda/targets/aarch64-linux/lib)
    include_directories(/usr/include/aarch64-linux-gnu/)
    link_directories(/usr/lib/aarch64-linux-gnu/)
else()
    message(STATUS "x86_64 or other platform")
    include_directories(/usr/local/cuda/include)
    link_directories(/usr/local/cuda/lib64)
    include_directories(/workspace/shared/TensorRT-8.6.1.6/include)
    link_directories(/workspace/shared/TensorRT-8.6.1.6/lib)
endif()

# 加载子模块
add_subdirectory(yolo11)
add_subdirectory(yolop)

# 可选：顶层可执行文件 (可省略)
add_executable(multi_model main.cpp udp/src/udp_sender.cpp)
target_link_libraries(multi_model PRIVATE yolo11_lib yolop_lib ${OpenCV_LIBS} nvinfer cudart)
target_link_libraries(multi_model PRIVATE nlohmann_json::nlohmann_json)