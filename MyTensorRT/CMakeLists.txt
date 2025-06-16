cmake_minimum_required(VERSION 3.16)
project(yolo LANGUAGES CXX CUDA)

# 设置构建类型
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 设置 C++17 和 CUDA 编译标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 设置 CUDA 和 TensorRT 路径
set(CUDA_TOOLKIT_ROOT /usr/local/cuda-11.4)
set(TENSORRT_ROOT /usr)

# 查找库
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

# 包含头文件
include_directories(
    include
    ${CUDA_TOOLKIT_ROOT}/include
    ${TENSORRT_ROOT}/include/aarch64-linux-gnu
    ${OpenCV_INCLUDE_DIRS}
)

# 源文件列表
set(SOURCES
    src/main.cpp 
    src/BaseTRTInfer.cpp 
    src/YOLO.cpp 
    src/YOLOInfer.cpp 
    src/YOLOPostprocess.cpp
    src/YOLOPostprocess.cu 
    src/YOLOPreprocess.cpp 
    src/YOLOPreprocess.cu
    src/IModel.cpp  # ⚠️ 加上这个，防止 typeinfo 报错
)

# 添加可执行文件
add_executable(yolo ${SOURCES})
set_target_properties(yolo PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# 链接库
target_link_libraries(yolo
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    nvinfer
    nvinfer_plugin
    nvonnxparser
    cudart
    pthread
    dl
)

# 编译定义
target_compile_definitions(yolo PRIVATE NOMINMAX)

# 输出目录
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)