# yolop 模块



# 头文件

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

include_directories(${OpenCV_INCLUDE_DIRS})

# yolop 源文件

file(GLOB YOLOP_SRCS

    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/*.cu

)



# 插件库

cuda_add_library(yolop_plugins SHARED ${CMAKE_CURRENT_SOURCE_DIR}/yololayer.cu OPTIONS -I${OpenCV_INCLUDE_DIRS})

target_link_libraries(yolop_plugins nvinfer cudart)



# yolop 主库

add_library(yolop_lib STATIC ${YOLOP_SRCS})

target_link_libraries(yolop_lib PUBLIC yolop_plugins nvinfer cudart ${OpenCV_LIBS})



# yolop demo 可执行文件

add_executable(yolop_det ${CMAKE_CURRENT_SOURCE_DIR}/yolop.cpp)

target_link_libraries(yolop_det PRIVATE yolop_lib)
