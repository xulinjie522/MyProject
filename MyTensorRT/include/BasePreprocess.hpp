#pragma once

#include <map>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"

class BasePreprocess{
public:
    virtual void preprocess(cv::Mat& img, float*, cudaStream_t) = 0;
    virtual ~BasePreprocess() = default;
};