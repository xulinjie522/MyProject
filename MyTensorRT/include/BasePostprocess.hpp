#pragma once
#include <NvInfer.h>
#include <opencv2/opencv.hpp>


class BasePostprocess{
public:
    virtual void doPlotResults(const cv::Mat& img) { PlotResults(img); }
private:
    virtual void PlotResults(const cv::Mat& img);
};