# TensorRT for Multi-model
本代码使用TensorRT同时加速yolop和yolo11模型，并部署在Nvidia AGX Orin设备上
## Test Environment
1. TensorRT 8.x
2. Nvidia AGX Orin
3. CUDA 11.4
## how to use
1. build yolop-yolo11
'<cd yolop-yolo11
   mkdir build && cd build
   cmake ..
   make>'
2. run
'<./multi-model >'
