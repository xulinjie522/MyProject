#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "yolo11/include/cuda_utils.h"
#include "yolo11/include/logging.h"
#include "yolo11/include/model.h"
#include "yolo11/include/postprocess.h"
#include "yolo11/include/preprocess.h"
#include "yolo11/include/utils.h"
#include "yolop/yolop.hpp"
#include "udp_sender.h"
#include <thread>
#include <mutex>
#include "processData.h"
#include "MultiCameraSync.h"

using namespace nvinfer1;

// YOLO11 配置
const int kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;

// yolop 配置
const int yolop_output_size = OUTPUT_SIZE;

void serialize_engine(std::string& wts_name, std::string& engine_name, float& gd, float& gw, int& max_channels,
                      std::string& type) {
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();
    IHostMemory* serialized_engine = nullptr;

    serialized_engine = buildEngineYolo11Det(builder, config, DataType::kFLOAT, wts_name, gd, gw, max_channels, type);

    assert(serialized_engine);
    std::ofstream p(engine_name, std::ios::binary);
    if (!p) {
        std::cout << "could not open plan output file" << std::endl;
        assert(false);
    }
    p.write(reinterpret_cast<const char*>(serialized_engine->data()), serialized_engine->size());

    delete serialized_engine;
    delete config;
    delete builder;
}

void deserialize_engine(std::string& engine_name, IRuntime** runtime, ICudaEngine** engine,
                        IExecutionContext** context) {
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
        assert(false);
    }
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    char* serialized_engine = new char[size];
    assert(serialized_engine);
    file.read(serialized_engine, size);
    file.close();

    *runtime = createInferRuntime(gLogger);
    assert(*runtime);
    *engine = (*runtime)->deserializeCudaEngine(serialized_engine, size);
    assert(*engine);
    *context = (*engine)->createExecutionContext();
    assert(*context);
    delete[] serialized_engine;
}

void prepare_buffer(ICudaEngine* engine, float** input_buffer_device, float** output_buffer_device,
                    float** output_buffer_host, float** decode_ptr_host, float** decode_ptr_device) {
    assert(engine->getNbBindings() == 2);
    const int inputIndex = engine->getBindingIndex(kInputTensorName);
    const int outputIndex = engine->getBindingIndex(kOutputTensorName);
    assert(inputIndex == 0);
    assert(outputIndex == 1);

    CUDA_CHECK(cudaMalloc((void**)input_buffer_device, kBatchSize * 3 * kInputH * kInputW * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)output_buffer_device, kBatchSize * kOutputSize * sizeof(float)));

    if (kBatchSize > 1) {
        std::cerr << "Do not yet support GPU post processing for multiple batches" << std::endl;
        exit(0);
    }

    *decode_ptr_host = new float[1 + kMaxNumOutputBbox * bbox_element];
    CUDA_CHECK(cudaMalloc((void**)decode_ptr_device, sizeof(float) * (1 + kMaxNumOutputBbox * bbox_element)));
}

void infer(IExecutionContext& context, cudaStream_t& stream, void** buffers, float* output,
           float* decode_ptr_host, float* decode_ptr_device, int model_bboxes) {

    context.enqueueV2(buffers, stream, nullptr);

    CUDA_CHECK(cudaMemsetAsync(decode_ptr_device, 0, sizeof(float) * (1 + kMaxNumOutputBbox * bbox_element), stream));
    cuda_decode((float*)buffers[1], model_bboxes, kConfThresh, decode_ptr_device, kMaxNumOutputBbox, stream);
    cuda_nms(decode_ptr_device, kNmsThresh, kMaxNumOutputBbox, stream);
    CUDA_CHECK(cudaMemcpyAsync(decode_ptr_host, decode_ptr_device,
                                sizeof(float) * (1 + kMaxNumOutputBbox * bbox_element), cudaMemcpyDeviceToHost,
                                stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));
}

int main(int argc, char** argv) {

    cudaSetDevice(kGpuId);  // YOLO11 用 kGpuId，YoloP 用 DEVICE，保持一致

    // 1️⃣ 加载 YOLO11 engine
    std::string yolo11_engine_name = "/home/nvidia/tensorrtx/tensorrtx/yolo11/build/yolo11.engine";
    IRuntime* yolo11_runtime = nullptr;
    ICudaEngine* yolo11_engine = nullptr;
    IExecutionContext* yolo11_context = nullptr;
    deserialize_engine(yolo11_engine_name, &yolo11_runtime, &yolo11_engine, &yolo11_context);

    cudaStream_t yolo11_stream;
    CUDA_CHECK(cudaStreamCreate(&yolo11_stream));
    cuda_preprocess_init(kMaxInputImageSize);
    auto yolo11_out_dims = yolo11_engine->getBindingDimensions(1);
    int yolo11_model_bboxes = yolo11_out_dims.d[0];

    float* yolo11_device_buffers[2];
    float* yolo11_output_buffer_host = nullptr;
    float* yolo11_decode_ptr_host = nullptr;
    float* yolo11_decode_ptr_device = nullptr;

    prepare_buffer(yolo11_engine, &yolo11_device_buffers[0], &yolo11_device_buffers[1],
                   &yolo11_output_buffer_host, &yolo11_decode_ptr_host, &yolo11_decode_ptr_device);

    // 2️⃣ 加载 YOLOP engine
    std::string yolop_engine_name = "/home/nvidia/tensorrtx/tensorrtx/yolop/build/yolop.engine";
    std::ifstream file(yolop_engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << yolop_engine_name << " error!" << std::endl;
        return -1;
    }
    char* yolop_trtModelStream = nullptr;
    size_t yolop_size = 0;
    file.seekg(0, file.end);
    yolop_size = file.tellg();
    file.seekg(0, file.beg);
    yolop_trtModelStream = new char[yolop_size];
    file.read(yolop_trtModelStream, yolop_size);
    file.close();

    IRuntime* yolop_runtime = createInferRuntime(gLogger);
    ICudaEngine* yolop_engine = yolop_runtime->deserializeCudaEngine(yolop_trtModelStream, yolop_size);
    IExecutionContext* yolop_context = yolop_engine->createExecutionContext();
    delete[] yolop_trtModelStream;

    void* yolop_buffers[4];
    const int yolop_inputIndex = yolop_engine->getBindingIndex(INPUT_BLOB_NAME);
    const int yolop_output_det_index = yolop_engine->getBindingIndex(OUTPUT_DET_NAME);
    const int yolop_output_seg_index = yolop_engine->getBindingIndex(OUTPUT_SEG_NAME);
    const int yolop_output_lane_index = yolop_engine->getBindingIndex(OUTPUT_LANE_NAME);
    CUDA_CHECK(cudaMalloc(&yolop_buffers[yolop_inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&yolop_buffers[yolop_output_det_index], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&yolop_buffers[yolop_output_seg_index], BATCH_SIZE * IMG_H * IMG_W * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&yolop_buffers[yolop_output_lane_index], BATCH_SIZE * IMG_H * IMG_W * sizeof(int)));

    cudaStream_t yolop_stream;
    CUDA_CHECK(cudaStreamCreate(&yolop_stream));

    static float yolop_data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
    static float yolop_prob[BATCH_SIZE * OUTPUT_SIZE];
    static int yolop_seg_out[BATCH_SIZE * IMG_H * IMG_W];
    static int yolop_lane_out[BATCH_SIZE * IMG_H * IMG_W];
    cv::Mat yolop_tmp_seg(IMG_H, IMG_W, CV_32S, yolop_seg_out);
    cv::Mat yolop_tmp_lane(IMG_H, IMG_W, CV_32S, yolop_lane_out);

    std::vector<cv::Vec3b> laneColor;
    laneColor.push_back(cv::Vec3b(0, 0, 0));
    laneColor.push_back(cv::Vec3b(0, 0, 255));

    //Create UDPSender
    UdpSender udp("192.168.10.10", 9090);

    //Save processed lane and object data
    std::map<int, std::vector<std::vector<cv::Point2d>>> allContours;
    std::map<int, std::vector<cv::Point2d>> allObject;

    //processData
    ProcessData processData;

    // 3️⃣ 打开摄像头
    MultiCameraSync cam_sync(1);
    try{
        if(!cam_sync.start()){
            throw std::runtime_error("Failed to start camera sync!");
        }
    }catch(const std::exception& e){
        std::cerr << "Standard exception: " << e.what() << std::endl;
        return -1;
    }catch(...){
        std::cerr << "Unknow exception occured!" << std::endl;
        return -1;
    }

    std::vector<cv::Mat> synced_frames;
    std::vector<cv::Mat> processed_frames;
    cv::Mat img;
    while (true) {
        allContours.clear();
        allObject.clear();
        auto start = std::chrono::system_clock::now();
        if(cam_sync.getSyncedFrames(synced_frames)){
            processed_frames.clear();
            for(const auto& frame : synced_frames){
                frame.copyTo(img);
                if (img.empty()) {
                    std::cerr << "No image!" << std::endl;
                    continue;
                }

                // ▶ YOLO11
                cuda_batch_preprocess(img, yolo11_device_buffers[0], kInputW, kInputH, yolo11_stream);

                // ▶ YOLOP
                cv::Mat pr_img = yolop::preprocess_img(img, INPUT_W, INPUT_H);
                //OpenMP
                #pragma omp parallel for collapse(2)
                for(int row = 0; row < INPUT_H; ++row){
                    for(int col = 0; col < INPUT_W; ++col){
                        int idx = row * INPUT_W + col;
                        float* uc_pixel = pr_img.ptr<float>(row) + col * 3;
                        yolop_data[idx] = uc_pixel[0];
                        yolop_data[idx + INPUT_H * INPUT_W] = uc_pixel[1];
                        yolop_data[idx + 2 * INPUT_H * INPUT_W] = uc_pixel[2];
                    }
                }
                
                //inference
                infer(*yolo11_context, yolo11_stream, (void**)yolo11_device_buffers, yolo11_output_buffer_host,
                    yolo11_decode_ptr_host, yolo11_decode_ptr_device, yolo11_model_bboxes);
                
                doInferenceCpu(*yolop_context, yolop_stream, yolop_buffers, yolop_data, yolop_prob,
                            yolop_seg_out, yolop_lane_out);

                cudaStreamSynchronize(yolo11_stream);
                cudaStreamSynchronize(yolop_stream);

                //postprocess
                std::vector<Detection> yolo11_res;
                process(yolo11_res, yolo11_decode_ptr_host, bbox_element, img);
                draw_bbox(img, yolo11_res, allObject, 0);   

                std::vector<Yolo::Detection> yolop_res;
                nms(yolop_res, yolop_prob, CONF_THRESH, NMS_THRESH);

                cv::Mat lane_res(img.rows, img.cols, CV_32S);
                cv::resize(yolop_tmp_lane, lane_res, lane_res.size(), 0, 0, cv::INTER_NEAREST);

                //OpenMP
                #pragma omp parallel for
                for(int row = 0; row < img.rows; ++row){
                    uchar* pdata = img.data + row * img.step;
                    for(int col = 0; col < img.cols; ++col){
                        int lane_idx = lane_res.at<int>(row, col);

                        for(int i = 0; i < 3; ++i){
                            if(lane_idx){
                                if(i != 2) {
                                    pdata[i] = pdata[i] / 2 + laneColor[lane_idx][i] / 2;
                                }
                            }
                        }
                        pdata += 3;
                    }
                }

                processed_frames.push_back(img);

                //processData
                auto unique_objects = processData.processObjects(allObject, 0.5);
                auto unique_lanes = processData.processLanes(lane_res, allContours, 0, 0.5);    

                //udp-sent
                json j = udp.packData(unique_objects, unique_lanes);
                udp.sendData(j);
            }
            // 4️⃣ 展示
            cv::Mat display;
            std::vector<cv::Mat> row1, row2;
            for(int i = 0; i < 3; ++i){
                cv::Mat resized;
                cv::resize(processed_frames[0], resized, cv::Size(640, 340));
                row1.push_back(resized);
            }
            for(int i = 3; i < 6; ++i){
                cv::Mat resized;
                cv::resize(processed_frames[0], resized, cv::Size(640, 340));
                row2.push_back(resized);
            }
            cv::Mat row1_cat, row2_cat;
            cv::hconcat(row1, row1_cat);
            cv::hconcat(row2, row2_cat);
            cv::vconcat(row1_cat, row2_cat, display);
            cv::resize(display, display, cv::Size(1280, 720));
            cv::imshow("Multi-Model", display);
            if (cv::waitKey(1) == 27) break;  // 按 ESC 退出
        }else{
            std::cerr << "Failed to getSyncedFrames!" << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(5));
        }

        auto end = std::chrono::system_clock::now();
        std::cout << "inference time: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    }

    // 5️⃣ 释放资源
    cudaStreamDestroy(yolo11_stream);
    cudaStreamDestroy(yolop_stream);
    CUDA_CHECK(cudaFree(yolo11_device_buffers[0]));
    CUDA_CHECK(cudaFree(yolo11_device_buffers[1]));
    CUDA_CHECK(cudaFree(yolo11_decode_ptr_device));
    delete[] yolo11_decode_ptr_host;
    delete[] yolo11_output_buffer_host;
    CUDA_CHECK(cudaFree(yolop_buffers[yolop_inputIndex]));
    CUDA_CHECK(cudaFree(yolop_buffers[yolop_output_det_index]));
    CUDA_CHECK(cudaFree(yolop_buffers[yolop_output_seg_index]));
    CUDA_CHECK(cudaFree(yolop_buffers[yolop_output_lane_index]));
    cuda_preprocess_destroy();
    yolo11_context->destroy();
    yolo11_engine->destroy();
    yolo11_runtime->destroy();
    yolop_context->destroy();
    yolop_engine->destroy();
    yolop_runtime->destroy();

    return 0;
}