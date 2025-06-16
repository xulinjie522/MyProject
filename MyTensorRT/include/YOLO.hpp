#pragma once

#include "IModel.hpp"
#include "YOLOPreprocess.hpp"
#include "YOLOPostprocess.hpp"
#include "YOLOInfer.hpp"
#include "Config.hpp"


class YOLO : public IModel{
public:
    YOLO(const Config& cfg) : config(cfg), Preprocess(cfg), Yoloinfer(cfg), Postprocess(cfg) {
        Yoloinfer.doLoadEngine();
    }
    
    void run(cv::Mat& img) override;
    ~YOLO() {}

private:
    std::vector<Detection> yolo_res;
    Config config;
    YOLOPreprocess Preprocess;
    YOLOInfer Yoloinfer;
    YOLOPostprocess Postprocess;
};