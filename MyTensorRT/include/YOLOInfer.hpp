#pragma once
#include "BaseTRTInfer.hpp"

using namespace nvinfer1;


class YOLOInfer : public BaseTRTInfer{
public:
    YOLOInfer(const Config& cfg) : BaseTRTInfer(cfg) {}
    ~YOLOInfer() = default;

    int getModel_bboxes(){ return model_bboxes; }
    float* getDecode_ptr_host(){ return decode_ptr_host;}
    float* getDecode_ptr_device(){ return deocde_ptr_device;}
private:
    void loadEngine() override;
    void allocateMemory() override;

private:
    float* decode_ptr_host;
    float* deocde_ptr_device;
    int model_bboxes;
};