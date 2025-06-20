#include "processData.h"

ProcessData::ProcessData(){
    std::cout << "Initializing Matrix!" << std::endl;

    homographyMatrices[0] = (cv::Mat_<double>(3, 3) <<
                            1.0, 0.0, 0.0,
                            0.0, 1.0, 0.0,
                            0.0, 0.0, 1.0);
    homographyMatrices[1] = (cv::Mat_<double>(3, 3) <<
                            1.0, 0.0, 0.0,
                            0.0, 1.0, 0.0,
                            0.0, 0.0, 1.0);
    homographyMatrices[2] = (cv::Mat_<double>(3, 3) <<
                            1.0, 0.0, 0.0,
                            0.0, 1.0, 0.0,
                            0.0, 0.0, 1.0);
    homographyMatrices[3] = (cv::Mat_<double>(3, 3) <<
                            1.0, 0.0, 0.0,
                            0.0, 1.0, 0.0,
                            0.0, 0.0, 1.0);
    homographyMatrices[4] = (cv::Mat_<double>(3, 3) <<
                            1.0, 0.0, 0.0,
                            0.0, 1.0, 0.0,
                            0.0, 0.0, 1.0);
    homographyMatrices[5] = (cv::Mat_<double>(3, 3) <<
                            1.0, 0.0, 0.0,
                            0.0, 1.0, 0.0,
                            0.0, 0.0, 1.0);
    std::cout << "Finish Init!" << std::endl;
}

std::vector<cv::Point2d>  ProcessData::processDet(const std::map<int, std::vector<cv::Point2d>>& allObject){
    std::vector<cv::Point2d> transformed_objects;
    for(auto it = allObject.begin(); it != allObject.end(); ++it){
        cv::Mat H = homographyMatrices[it->first];

        for(const cv::Point& pt : it->second){
            std::vector<cv::Point2d> src_points({pt});
            std::vector<cv::Point2d> dst_points;
            cv::perspectiveTransform(src_points, dst_points, H);
            if(!dst_points.empty()){
                transformed_objects.push_back(dst_points[0]);
            }
        }
    }
    return transformed_objects;
}

std::vector<cv::Point2d> ProcessData::filterObjects(const std::vector<cv::Point2d>& Objects, const double& threshold){
    std::vector<cv::Point2d> det_res;
    det_res.reserve(Objects.size());
    const double squared_threshold = threshold * threshold;
    for(const auto& object : Objects){
        bool flag = false;
        for(const auto& res : det_res){
            const double dx = object.x - res.x;
            const double dy = object.y - res.y;
            const double distance =  dx * dx + dy * dy;
            if(distance < squared_threshold){
                flag = true;
                break;
            }
        }
        if(!flag) det_res.emplace_back(object);
    }
    return det_res;
}

std::vector<cv::Point2d> ProcessData::processObjects(const std::map<int, std::vector<cv::Point2d>>& allObject, const double& threshold){
    if(allObject.empty()){
        std::cout << "No object detected in the current frame!" << std::endl;
        return {};
    }
    auto transformedObjects = processDet(allObject);
    return filterObjects(transformedObjects, threshold);
}

std::vector<std::vector<cv::Point2d>> 
ProcessData::processLane(cv::Mat& lane_res, std::map<int, std::vector<std::vector<cv::Point2d>>>& allContours, int cam_id){
    // 创建二值图像，仅保留车道线部分（lane_res 中非零部分）
    cv::Mat binary_mask = (lane_res > 0);
    std::vector<std::vector<cv::Point>> contours;
    std::vector<std::vector<cv::Point2d>> contours2d;

    // 提取所有轮廓
    cv::findContours(binary_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    //显示提取的轮廓
    // cv::Mat contourImage = cv::Mat::zeros(binary_mask.size(), CV_8UC3);
    // cv::drawContours(contourImage, contours, -1, {0, 255, 0}, 2);
    // cv::imshow("Contours", contourImage);
    // cv::waitKey(0);
    if(contours.size() != 0)
    {
        for(const auto& contour : contours){
            std::vector<cv::Point2d> contour2d;
            for(const auto& point : contour){
                contour2d.push_back(cv::Point2d(point.x, point.y));
            }
            contours2d.push_back(contour2d);
        }
        allContours[cam_id] = contours2d;
    }else{
        std::cout << "No lane detected in the current frame!" << std::endl;
    }

    std::vector<std::vector<cv::Point2d>> transformedContours;
    for(const auto& it : allContours){
        for(const auto& contour : it.second){
            std::vector<cv::Point2d> transformedContour;
            cv::Mat H = homographyMatrices[it.first];
            cv::perspectiveTransform(contour, transformedContour, H);
            transformedContours.emplace_back(transformedContour);
        }
    }
    return transformedContours;
}

std::vector<cv::Vec3f> ProcessData::fitQuadraticCurve(const std::vector<std::vector<cv::Point2d>>& points){
    std::vector<cv::Vec3f> res;
    for(auto point : points){
        int n = point.size();

        if (n < 3) {
            std::cerr << "At least 3 points are required for quadratic fitting." << std::endl;
            return {};
        }

        // 构造矩阵 A 和 b
        cv::Mat A(n, 3, CV_64F); // 设计矩阵 [y^2, y, 1]
        cv::Mat b(n, 1, CV_64F); // 响应矩阵 x

        for (int i = 0; i < n; ++i) {
            double y = point[i].y;
            A.at<double>(i, 0) = y * y; // y^2
            A.at<double>(i, 1) = y;     // y
            A.at<double>(i, 2) = 1.0;   // 常数项
            b.at<double>(i, 0) = point[i].x; // x
        }

        // 使用最小二乘法求解 A * [A B C]^T = b
        cv::Mat coeffs;
        cv::solve(A, b, coeffs, cv::DECOMP_SVD); // 解线性方程组

        // 返回拟合系数 A, B, C
        res.push_back(cv::Vec3f(coeffs.at<double>(0, 0), coeffs.at<double>(1, 0), coeffs.at<double>(2, 0)));       
    }
    return res;
}

std::vector<cv::Vec3f> 
ProcessData::filterLanes(const std::vector<cv::Vec3f>& allCoefficients, const double& threshold){
    const double squared_threshold = threshold * threshold;
    std::vector<cv::Vec3f> uniqueLanes;
    uniqueLanes.reserve(allCoefficients.size());
    for(const auto& coeffs : allCoefficients){
        bool flag = false;
        for(const auto& uniqueLane : uniqueLanes){
            const double diffA = coeffs[0] - uniqueLane[0];
            const double diffB = coeffs[1] - uniqueLane[1];
            const double diffC = coeffs[2] - uniqueLane[2];
            if(diffA * diffA + diffB * diffB + diffC * diffC < squared_threshold){
                flag = true;
                break;
            }
        }
        if(!flag) uniqueLanes.emplace_back(coeffs);
    }
    return uniqueLanes;
}

std::vector<cv::Vec3f> ProcessData::processLanes(cv::Mat& yolop_res, 
        std::map<int, std::vector<std::vector<cv::Point2d>>>& allContours, int cam_id, const double& threshold){
    
    auto transformedContours = processLane(yolop_res, allContours, cam_id);
    auto coefficients = fitQuadraticCurve(transformedContours);
    auto uniqueLanes = filterLanes(coefficients, threshold);
    return uniqueLanes;
}