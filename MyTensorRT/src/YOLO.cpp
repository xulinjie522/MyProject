#include "YOLO.hpp"

void YOLO::run(cv::Mat& img){
    Preprocess.preprocess(img, Yoloinfer.getBuffers()[0], Yoloinfer.getStream());
    Yoloinfer.doInfer();
    Postprocess.postprocess(Yoloinfer.getStream(), Yoloinfer.getDecode_ptr_host(), 
                            Yoloinfer.getDecode_ptr_device(), Yoloinfer.getModel_bboxes(), yolo_res, img, Yoloinfer.getBuffers());                      
    Postprocess.doPlotResults(img, yolo_res);
}
