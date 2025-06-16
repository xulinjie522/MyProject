#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>


class IModel{

public:
    virtual void run(cv::Mat& img) = 0;
    virtual ~IModel();
};