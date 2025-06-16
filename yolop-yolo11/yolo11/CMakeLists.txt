# yolo11 模块



# 头文件

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/plugin)

include_directories(${OpenCV_INCLUDE_DIRS})

# 源文件

file(GLOB YOLO11_SRCS

    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu

    ${CMAKE_CURRENT_SOURCE_DIR}/plugin/*.cpp

)



# 插件库

cuda_add_library(yolo11_plugins SHARED ${CMAKE_CURRENT_SOURCE_DIR}/plugin/yololayer.cu OPTIONS -I${OpenCV_INCLUDE_DIRS})

target_link_libraries(yolo11_plugins nvinfer cudart)



# 主库

add_library(yolo11_lib STATIC ${YOLO11_SRCS})

target_link_libraries(yolo11_lib PUBLIC yolo11_plugins nvinfer cudart ${OpenCV_LIBS})



# demo 可执行文件

add_executable(yolo11_det ${CMAKE_CURRENT_SOURCE_DIR}/yolo11_det.cpp)

target_link_libraries(yolo11_det PRIVATE yolo11_lib)