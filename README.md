# TensorRT for Multi-model
## notice
The original code is from [wang-xinyu/tensorrtx](https://github.com/wang-xinyu/tensorrtx)
本代码使用TensorRT同时加速yolop和yolo11模型，并部署在Nvidia AGX Orin设备上
## Test Environment
1. TensorRT 8.x
2. Nvidia AGX Orin
3. CUDA 11.4
## how to use
1. build yolop-yolo11
```
   cd yolop-yolo11
   mkdir build && cd build
   cmake ..
   make
```
3. run
```
   ./multi-model
```
## Acknowledgments
This project incorporates code from the following third-party source:
- **Source**: [WangXinyu/original-repo](https://github.com/WangXinyu/original-repo)
- **License**: MIT License
- **Copyright**: © 2019-2020 Wang Xinyu
- **Usage**: [Implement multi-task detection of lane lines and targets]
