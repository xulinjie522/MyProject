#include <iostream>
#include <opencv2/opencv.hpp>
#include "YOLO.hpp"

int main()
{
    Config cfg;
    cfg.enginePath = "/home/nvidia/Desktop/xulinjie/yolo11.trt";
    YOLO yolo(cfg);

    cv::VideoCapture cap(0, cv::CAP_GSTREAMER);
    if(!cap.isOpened()){
        std::cerr << "Open camera failed!" << std::endl;
        return -1;
    }

    cv::Mat frame;
    while(true){
        cap >> frame;

        if(frame.empty()){
            std::cerr << "there is no img!" << std::endl;
            continue;
        }

        yolo.run(frame);
        // cv::imwrite("/home/nvidia/Desktop/xulinjie/detect-res/img.jpg", frame);
        cv::imshow("detect", frame);
        cv::waitKey(1);
    }


    return 0;
}