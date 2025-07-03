#include "YOLOPostprocess.hpp"
#include "utils.hpp"

cv::Rect YOLOPostprocess::get_rect(cv::Mat& img, float bbox[4]) {
    float l, r, t, b;
    float r_w = cfg.kInputW / (img.cols * 1.0);
    float r_h = cfg.kInputH / (img.rows * 1.0);

    if (r_h > r_w) {
        l = bbox[0];
        r = bbox[2];
        t = bbox[1] - (cfg.kInputH - r_w * img.rows) / 2;
        b = bbox[3] - (cfg.kInputH - r_w * img.rows) / 2;
        l = l / r_w;
        r = r / r_w;
        t = t / r_w;
        b = b / r_w;
    } else {
        l = bbox[0] - (cfg.kInputW - r_h * img.cols) / 2;
        r = bbox[2] - (cfg.kInputW - r_h * img.cols) / 2;
        t = bbox[1];
        b = bbox[3];
        l = l / r_h;
        r = r / r_h;
        t = t / r_h;
        b = b / r_h;
    }
    l = std::max(0.0f, l);
    t = std::max(0.0f, t);
    int width = std::max(0, std::min(int(round(r - l)), img.cols - int(round(l))));
    int height = std::max(0, std::min(int(round(b - t)), img.rows - int(round(t))));

    return cv::Rect(int(round(l)), int(round(t)), width, height);
}

static float iou(float lbox[4], float rbox[4]) {
    float interBox[] = {
            (std::max)(lbox[0], rbox[0]),
            (std::min)(lbox[2], rbox[2]),
            (std::max)(lbox[1], rbox[1]),
            (std::min)(lbox[3], rbox[3]),
    };

    if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS = (interBox[1] - interBox[0]) * (interBox[3] - interBox[2]);
    float unionBoxS = (lbox[2] - lbox[0]) * (lbox[3] - lbox[1]) + (rbox[2] - rbox[0]) * (rbox[3] - rbox[1]) - interBoxS;
    return interBoxS / unionBoxS;
}

static bool cmp(const Detection& a, const Detection& b) {
    if (a.conf == b.conf) {
        return a.bbox[0] < b.bbox[0];
    }
    return a.conf > b.conf;
}


void process_decode_ptr_host(std::vector<Detection>& res, const float* decode_ptr_host, int bbox_element, cv::Mat& img,
                             int count) {
    Detection det;
    for (int i = 0; i < count; i++) {
        int basic_pos = 1 + i * bbox_element;
        int keep_flag = decode_ptr_host[basic_pos + 6];
        if (keep_flag == 1) {
            det.bbox[0] = decode_ptr_host[basic_pos + 0];
            det.bbox[1] = decode_ptr_host[basic_pos + 1];
            det.bbox[2] = decode_ptr_host[basic_pos + 2];
            det.bbox[3] = decode_ptr_host[basic_pos + 3];
            det.conf = decode_ptr_host[basic_pos + 4];
            det.class_id = decode_ptr_host[basic_pos + 5];
            res.push_back(det);
        }
    }
}

void YOLOPostprocess::batch_process(std::vector<Detection>& res_batch, const float* decode_ptr_host,
                   int bbox_element, cv::Mat& img) {
    int count = static_cast<int>(*decode_ptr_host);
    count = std::min(count, cfg.kMaxNumOutputBbox);
    process_decode_ptr_host(res_batch, decode_ptr_host, bbox_element, img, count);
}

void YOLOPostprocess::draw_bbox(cv::Mat& img, std::vector<Detection>& res) {
        for (size_t j = 0; j < res.size(); j++) {
            cv::Rect r = get_rect(img, res[j].bbox);
            cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
            cv::putText(img, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2,
                        cv::Scalar(0xFF, 0xFF, 0xFF), 2);
        }
}

void YOLOPostprocess::nms(std::vector<Detection>& res, float* output, float conf_thresh, float nms_thresh) {
    int det_size = sizeof(Detection) / sizeof(float);
    std::map<float, std::vector<Detection>> m;

    for (int i = 0; i < output[0]; i++) {
        if (output[1 + det_size * i + 4] <= conf_thresh || isnan(output[1 + det_size * i + 4]))
            continue;
        Detection det;
        memcpy(&det, &output[1 + det_size * i], det_size * sizeof(float));
        if (m.count(det.class_id) == 0)
            m.emplace(det.class_id, std::vector<Detection>());
        m[det.class_id].push_back(det);
    }
    for (auto it = m.begin(); it != m.end(); it++) {
        auto& dets = it->second;
        std::sort(dets.begin(), dets.end(), cmp);
        for (size_t m = 0; m < dets.size(); ++m) {
            auto& item = dets[m];
            res.push_back(item);
            for (size_t n = m + 1; n < dets.size(); ++n) {
                if (iou(item.bbox, dets[n].bbox) > nms_thresh) {
                    dets.erase(dets.begin() + n);
                    --n;
                }
            }
        }
    }
}

void YOLOPostprocess::postprocess(cudaStream_t& stream, float* decode_ptr_host, float* decode_ptr_device, int model_bboxes, std::vector<Detection>& res, cv::Mat& img, float** buffers){
    CHECK(
        cudaMemsetAsync(decode_ptr_device, 0, sizeof(float) * (1 + cfg.kMaxNumOutputBbox * bbox_element), stream));
    cuda_decode(buffers[1], model_bboxes, cfg.kConfThresh, decode_ptr_device, cfg.kMaxNumOutputBbox, stream);
    cuda_nms(decode_ptr_device, cfg.kNmsThresh, cfg.kMaxNumOutputBbox, stream);  //cuda nms
    CHECK(cudaMemcpyAsync(decode_ptr_host, decode_ptr_device,
                            sizeof(float) * (1 + cfg.kMaxNumOutputBbox * bbox_element), cudaMemcpyDeviceToHost,
                            stream));
    CHECK(cudaStreamSynchronize(stream));
    batch_process(res, decode_ptr_host, bbox_element, img);
}

void YOLOPostprocess::PlotResults(cv::Mat& img, std::vector<Detection>& yolo_res){
    std::cout << "yolo_res.size(): " << yolo_res.size() << std::endl;
    for(const auto& det : yolo_res){
        std::cout << "bbox: [" << det.bbox[0] << ", " << det.bbox[1] << ", " << det.bbox[2] << ", " << det.bbox[3] << "] " << ", class_id: " << det.class_id << ", conf: " << det.conf << std::endl;
    }
    draw_bbox(img, yolo_res);
}

void YOLOPostprocess::doPlotResults(cv::Mat& img, std::vector<Detection>& yolo_res){
    PlotResults(img, yolo_res);
}