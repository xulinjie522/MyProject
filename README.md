# TensorRT for Multi-model
## notice
The original code is from [wang-xinyu/tensorrtx](https://github.com/wang-xinyu/tensorrtx)  
This code utilizes TensorRT to simultaneously accelerate both YOLOP and YOLOv11 models, deploying them on NVIDIA AGX Orin devices.
## Test Environment
1. TensorRT 8.5.2
2. Nvidia AGX Orin(Jetson L4T R35.4.1)
3. CUDA 11.4
4. cuDNN 8.6.0
5. OpenCV 4.5.4
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
