#include "MultiCameraSync.h"


MultiCameraSync::MultiCameraSync(int cam_count)
    : camera_count_(cam_count), running_(false)
{
    buffers_ = new std::deque<FrameData>[camera_count_];
    mutexes_ = new std::mutex[camera_count_];
}

MultiCameraSync::~MultiCameraSync(){
    stop();
    delete[] buffers_;
    delete[] mutexes_;
}

bool MultiCameraSync::start(){
    running_ = true;
    for(int i = 0; i < camera_count_; ++i){
        threads_.emplace_back(&MultiCameraSync::captureThread, this, i, i);
    }
    std::cout << "Starting camera_thread!" << std::endl;
    return true;
}

void MultiCameraSync::stop(){
    running_ = false;
    for(auto& t : threads_){
        if(t.joinable()) t.join();
    }
    threads_.clear();
}

void MultiCameraSync::captureThread(int cam_idx, int device_id){
    cv::VideoCapture cap(device_id);
    if(!cap.isOpened()){
        std::cerr << "Falied to open camera!" << std::endl;
        return;
    }else{
        std::cout << "Sucess to open camera: " << device_id << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(10));
    }

    // cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    // cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    // cap.set(cv::CAP_PROP_FRAME_HEIGHT, 640);

    cv::Mat frame;
    while(running_){
        cap >> frame;
        if(frame.empty()) continue;

        FrameData data;
        data.frame = frame;
        data.timestamp = std::chrono::steady_clock::now();

        std::lock_guard<std::mutex> lock(mutexes_[cam_idx]);
        buffers_[cam_idx].push_back(data);
        if(buffers_[cam_idx].size() > 10){
            buffers_[cam_idx].pop_front();
        }
    }
}

bool MultiCameraSync::getSyncedFrames(std::vector<cv::Mat>& synced_frames){
    synced_frames.resize(camera_count_);
    std::vector<std::deque<FrameData>> snapshots(camera_count_);

    for(int i = 0; i < camera_count_; ++i){
        std::lock_guard<std::mutex> lock(mutexes_[i]);
        if(buffers_[i].empty()) return false;
        snapshots[i] = buffers_[i];
    }
    
    auto base_time = snapshots[0].back().timestamp;
    synced_frames[0] = snapshots[0].back().frame;

    for(int i = 1; i < camera_count_; ++i){
        auto& deque =snapshots[i];
        auto closest = deque.front();
        auto min_diff = std::abs(std::chrono::duration_cast<std::chrono::milliseconds>(
            base_time - closest.timestamp).count());

        for(auto& f : deque){
            auto diff = std::abs(std::chrono::duration_cast<std::chrono::milliseconds>(
                base_time - f.timestamp).count());
                if(diff < min_diff){
                    min_diff = diff;
                    closest = f;
                }
        }
        synced_frames[i] = closest.frame;
    }

    return true;
}