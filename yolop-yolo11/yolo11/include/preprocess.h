#pragma once

#include <map>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "types.h"

void cuda_preprocess_init(int max_image_size);

void cuda_preprocess_destroy();

void cuda_preprocess(uint8_t* src, int src_width, int src_height, float* dst, int dst_width, int dst_height,
                     cudaStream_t stream);

void cuda_batch_preprocess(cv::Mat& img, float* dst, int dst_width, int dst_height,
                           cudaStream_t stream);
