#pragma once  // 防止头文件重复包含

#include <cstddef>  // 用于 size_t

struct alignas(float) Detection {
    //center_x center_y w h
    float bbox[4];
    float conf;  // bbox_conf * cls_conf
    float class_id;
};

struct Config {
    std::string enginePath;
    // Tensor names (字符串常量)
    static constexpr const char* kInputTensorName = "images";
    static constexpr const char* kOutputTensorName = "output";
    static constexpr const char* kProtoTensorName = "proto";

    // 模型参数 (整型常量)
    static constexpr int kNumClass = 80;
    static constexpr int kPoseNumClass = 1;
    static constexpr int kNumberOfPoints = 17;  // 关键点数量
    static constexpr int kObbNumClass = 15;      // OBB 模型的类别数
    static constexpr int kObbNe = 1;             // 额外参数数量
    static constexpr int kBatchSize = 1;
    static constexpr int kGpuId = 0;

    // 输入尺寸
    static constexpr int kInputH = 640;
    static constexpr int kInputW = 640;


    // 阈值参数 (浮点型常量)
    static constexpr float kNmsThresh = 0.45f;
    static constexpr float kConfThresh = 0.5f;
    static constexpr float kConfThreshKeypoints = 0.5f;  // 关键点置信度

    // 内存/缓冲区限制
    static constexpr size_t kMaxInputImageSize = 3000 * 3000;  // 使用 size_t 更合适
    static constexpr int kMaxNumOutputBbox = 1000;

    // 分类模型类别数
    static constexpr int kClsNumClass = 1000;

    // 量化校准数据路径 (建议改用 std::string_view 或 const std::string&)
    static constexpr const char* kInputQuantizationFolder = "./coco_calib";

    //prepare buffer
    static constexpr int InputSize = kBatchSize * 3 * kInputH * kInputW * sizeof(float);
    static constexpr int OutputSize = (kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1) * sizeof(float);
};

struct AffineMatrix {
    float value[6];
};

const int bbox_element =
        sizeof(AffineMatrix) / sizeof(float) + 1;  // left, top, right, bottom, confidence, class, keepflag