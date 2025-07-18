#pragma once

#include <dirent.h>
#include <opencv2/opencv.hpp>

#include <iostream>
#include "common.hpp"

#define SHOW_IMG

namespace yolop{
    static inline cv::Mat preprocess_img(cv::Mat& img, int input_w, int input_h) {
        int w, h, x, y;
        float r_w = input_w / (img.cols*1.0);
        float r_h = input_h / (img.rows*1.0);
        if (r_h > r_w) {
            w = input_w;
            h = r_w * img.rows;
            x = 0;
            y = (input_h - h) / 2;
        } else {
            w = r_h * img.cols;
            h = input_h;
            x = (input_w - w) / 2;
            y = 0;
        }
        cv::Mat re(h, w, CV_8UC3);
        cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
        cv::Mat out(input_h, input_w, CV_8UC3, cv::Scalar(114, 114, 114));
        re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
        cv::Mat tensor;
        out.convertTo(tensor, CV_32FC3, 1.f / 255.f);

        cv::subtract(tensor, cv::Scalar(0.485, 0.456, 0.406), tensor, cv::noArray(), -1);
        cv::divide(tensor, cv::Scalar(0.229, 0.224, 0.225), tensor, 1, -1);
        return tensor;
    }

    static inline int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names) {
        DIR *p_dir = opendir(p_dir_name);
        if (p_dir == nullptr) {
            return -1;
        }

        struct dirent* p_file = nullptr;
        while ((p_file = readdir(p_dir)) != nullptr) {
            if (strcmp(p_file->d_name, ".") != 0 && strcmp(p_file->d_name, "..") != 0) {
                std::string cur_file_name(p_file->d_name);
                file_names.push_back(cur_file_name);
            }
        }

        closedir(p_dir);
        return 0;
    }
}
