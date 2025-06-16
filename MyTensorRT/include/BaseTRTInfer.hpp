#pragma once
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include <chrono>
#include "BasePreprocess.hpp"
#include "Config.hpp"
#include "Logger.hpp"

using namespace nvinfer1;

#define CHECK(status)                                                        \
    do {                                                                     \
        auto ret = (status);                                                 \
        if (ret != cudaSuccess) {                                            \
            std::cerr << "CUDA error: " << cudaGetErrorString(ret)           \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)


class BaseTRTInfer{
public:
    BaseTRTInfer(const Config& cfg) : config(cfg), logger() {}
    void doLoadEngine();
    void doInfer();
    virtual ~BaseTRTInfer();
    float* getInputPtr();
    float* getOutputPtr();
    float** getBuffers(){ return buffers; }
    cudaStream_t& getStream() { return stream; }
private:
    void infer(float* input, float* output);
    virtual void loadEngine();
    virtual void allocateMemory() = 0;
protected:
    Config config;
    Logger logger;
    ICudaEngine* engine = nullptr;
    IExecutionContext* context = nullptr;
    float* buffers[2];
    cudaStream_t stream;
    int inputIndex, outputIndex;
    std::unique_ptr<float[]> hInput, hOutput;
};