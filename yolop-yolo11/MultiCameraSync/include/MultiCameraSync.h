#pragma once
#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <deque>
#include <vector>
#include <chrono>
#include <atomic>
#include <iostream>
#include <cmath>

struct FrameData{
    cv::Mat frame;
    std::chrono::steady_clock::time_point timestamp;
};

class MultiCameraSync{
public:
    MultiCameraSync(int cam_count = 6);
    ~MultiCameraSync();

    bool start();

    bool getSyncedFrames(std::vector<cv::Mat>& synced_frames);

    void stop();

private:
    void captureThread(int cam_idx, int device_id);

    int camera_count_;
    std::vector<std::thread> threads_;
    std::deque<FrameData>* buffers_;
    std::mutex* mutexes_;
    std::atomic<bool> running_;
};