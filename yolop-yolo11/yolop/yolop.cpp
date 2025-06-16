#include "yolop.hpp"


int main(int argc, char** argv) {
    cudaSetDevice(DEVICE);

    std::string engine_name = "/home/nvidia/tensorrtx/tensorrtx/yolop/build/yolop.engine";

    // deserialize the .engine and run inference
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
        return -1;
    }
    char *trtModelStream = nullptr;
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();


    // prepare input data ---------------------------
    static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
    static float prob[BATCH_SIZE * OUTPUT_SIZE];
    static int seg_out[BATCH_SIZE * IMG_H * IMG_W];
    static int lane_out[BATCH_SIZE * IMG_H * IMG_W];
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    assert(engine->getNbBindings() == 4);
    void* buffers[4];
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    const int output_det_index = engine->getBindingIndex(OUTPUT_DET_NAME);
    const int output_seg_index = engine->getBindingIndex(OUTPUT_SEG_NAME);
    const int output_lane_index = engine->getBindingIndex(OUTPUT_LANE_NAME);
    assert(inputIndex == 0);
    assert(output_det_index == 1);
    assert(output_seg_index == 2);
    assert(output_lane_index == 3);
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[output_det_index], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[output_seg_index], BATCH_SIZE * IMG_H * IMG_W * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&buffers[output_lane_index], BATCH_SIZE * IMG_H * IMG_W * sizeof(int)));
    // Create stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // store seg results
    cv::Mat tmp_seg(IMG_H, IMG_W, CV_32S, seg_out);
    // store lane results
    cv::Mat tmp_lane(IMG_H, IMG_W, CV_32S, lane_out);
    // PrintMat(tmp_seg);
    std::vector<cv::Vec3b> segColor;
    segColor.push_back(cv::Vec3b(0, 0, 0));
    segColor.push_back(cv::Vec3b(0, 255, 0));
    segColor.push_back(cv::Vec3b(255, 0, 0));

    std::vector<cv::Vec3b> laneColor;
    laneColor.push_back(cv::Vec3b(0, 0, 0));
    laneColor.push_back(cv::Vec3b(0, 0, 255));
    laneColor.push_back(cv::Vec3b(0, 0, 0));

    cv::VideoCapture cap(0, cv::CAP_GSTREAMER);
    if(!cap.isOpened()) std::cerr << "camera failed to open!" << std::endl;

    cv::Mat img;

    while(true){
        cap >> img;
        // preprocess ~3ms
        if (img.empty()) {
            std::cerr << "No image!" << std::endl;
            continue;
        }
        //cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        cv::Mat pr_img = yolop::preprocess_img(img, INPUT_W, INPUT_H); // letterbox
        int i = 0;
        // BGR to RGB and normalize
        for (int row = 0; row < INPUT_H; ++row) {
            float* uc_pixel = pr_img.ptr<float>(row);
            for (int col = 0; col < INPUT_W; ++col) {
                data[i] = uc_pixel[0];
                data[i + INPUT_H * INPUT_W] = uc_pixel[1];
                data[i + 2 * INPUT_H * INPUT_W] = uc_pixel[2];
                uc_pixel += 3;
                ++i;
            }
        }

        // Run inference
        auto start = std::chrono::system_clock::now();
        doInferenceCpu(*context, stream, buffers, data, prob, seg_out, lane_out);
        auto end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

        // postprocess ~0ms
        std::vector<Yolo::Detection> res;
        nms(res, prob, CONF_THRESH, NMS_THRESH);

        // show results
        // handling seg and lane results
        cv::Mat lane_res(img.rows, img.cols, CV_32S);
        cv::resize(tmp_lane, lane_res, lane_res.size(), 0, 0, cv::INTER_NEAREST);
        for (int row = 0; row < img.rows; ++row) {
            uchar* pdata = img.data + row * img.step;
            for (int col = 0; col < img.cols; ++col) {
                int lane_idx = lane_res.at<int>(row, col);
                for (int i = 0; i < 3; ++i) {
                    if (lane_idx) {
                        if (i != 2)
                            pdata[i] = pdata[i] / 2 + laneColor[lane_idx][i] / 2;
                    }
                }
                pdata += 3;
            }
        }
        cv::imshow("lane", img);
        cv::waitKey(1);
    }

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(buffers[inputIndex]));
    CUDA_CHECK(cudaFree(buffers[output_det_index]));
    CUDA_CHECK(cudaFree(buffers[output_seg_index]));
    CUDA_CHECK(cudaFree(buffers[output_lane_index]));
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    return 0;
}
