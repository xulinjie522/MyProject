#include "YOLOPreprocess.hpp"

void YOLOPreprocess::preprocess(cv::Mat& img, float* buffer, cudaStream_t stream){
    cv::resize(img, img, cv::Size(cfg.kInputW, cfg.kInputH));
    cuda_batch_preprocess(img, buffer, cfg.kInputW, cfg.kInputH, stream);
}