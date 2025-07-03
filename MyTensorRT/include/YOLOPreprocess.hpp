#pragma once

#include "BasePreprocess.hpp"
#include "Config.hpp"
#include "BaseTRTInfer.hpp"


class YOLOPreprocess : public BasePreprocess{
public:
    YOLOPreprocess(const Config& config) : cfg(config) {
        cuda_preprocess_init(config.kMaxInputImageSize);
    }
    void preprocess(cv::Mat& img, float*, cudaStream_t) override;
private:
    void cuda_preprocess_init(int max_image_size);

    void cuda_preprocess_destroy();

    void cuda_preprocess(uint8_t* src, int src_width, int src_height, float* dst, int dst_width, int dst_height,
                        cudaStream_t stream);

    void cuda_batch_preprocess(cv::Mat& img, float* dst, int dst_width, int dst_height,
                            cudaStream_t stream);
private:
    Config cfg;
    uint8_t* img_buffer_host = nullptr;
    uint8_t* img_buffer_device = nullptr;
};