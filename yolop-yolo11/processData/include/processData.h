#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include "/home/nvidia/Desktop/xulinjie/yolop-yolo11/yolo11/include/types.h"

class ProcessData{
public:
    ProcessData();
    std::vector<cv::Point2d> processObjects(const std::map<int, std::vector<cv::Point2d>>& allObject, const double& threshold);
    std::vector<cv::Vec3f> processLanes(cv::Mat& yolop_res, 
        std::map<int, std::vector<std::vector<cv::Point2d>>>& allContours, int cam_id, const double& threshold);

private:
    std::vector<cv::Point2d> processDet(const std::map<int, std::vector<cv::Point2d>>& allObject);
    std::vector<cv::Point2d> filterObjects(const std::vector<cv::Point2d>& Objects, const double& threshold);
    std::vector<std::vector<cv::Point2d>> processLane(cv::Mat& yolop_res, 
        std::map<int, std::vector<std::vector<cv::Point2d>>>& allContours, int cam_id);
    std::vector<cv::Vec3f> fitQuadraticCurve(const std::vector<std::vector<cv::Point2d>>& points);
    std::vector<cv::Vec3f> filterLanes(const std::vector<cv::Vec3f>& allCoefficients, const double& threshold);
private:
    cv::Mat homographyMatrices[6];
};