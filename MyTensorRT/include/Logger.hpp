#pragma once
#include <NvInfer.h>  // 包含 ILogger 的定义
#include <cuda_runtime_api.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>

// Logger for TensorRT
class Logger : public nvinfer1::ILogger {  // 必须使用完整命名空间
public:
    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override {
        switch (severity) {
            case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
                std::cerr << "[TensorRT INTERNAL_ERROR]: " << msg << std::endl;
                break;
            case nvinfer1::ILogger::Severity::kERROR:
                std::cerr << "[TensorRT ERROR]: " << msg << std::endl;
                break;
            case nvinfer1::ILogger::Severity::kWARNING:
                std::cerr << "[TensorRT WARNING]: " << msg << std::endl;
                break;
            case nvinfer1::ILogger::Severity::kINFO:
                std::cout << "[TensorRT INFO]: " << msg << std::endl;
                break;
            case nvinfer1::ILogger::Severity::kVERBOSE:
                std::cout << "[TensorRT VERBOSE]: " << msg << std::endl;
                break;
            default:
                std::cerr << "[TensorRT UNKNOWN]: " << msg << std::endl;
        }
    }
};