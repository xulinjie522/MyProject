#include "BasePostprocess.hpp"
#include "Config.hpp"
#include "BaseTRTInfer.hpp"

class YOLOPostprocess{
public:
    YOLOPostprocess() = default;
    YOLOPostprocess(const Config& config) : cfg(config) {}
    void postprocess(cudaStream_t& stream, float* decode_ptr_host, float* decode_ptr_device, int model_bboxes, std::vector<Detection>& res, cv::Mat& img, float** buffers);
    void doPlotResults(cv::Mat& img, std::vector<Detection>& yolo_res);
private:
    void batch_process(std::vector<Detection>& res_batch, const float* decode_ptr_host,
                   int bbox_element, cv::Mat& img);

    void cuda_decode(float* predict, int num_bboxes, float confidence_threshold, float* parray, int max_objects,
                 cudaStream_t stream);

    void cuda_nms(float* parray, float nms_threshold, int max_objects, cudaStream_t stream);

    void nms(std::vector<Detection>& res, float* output, float conf_thresh, float nms_thresh = 0.5);

    void PlotResults(cv::Mat& img, std::vector<Detection>& yolo_res);

    cv::Rect get_rect(cv::Mat& img, float bbox[4]);

    void draw_bbox(cv::Mat& img, std::vector<Detection>& res);
    
private:
    Config cfg;
};