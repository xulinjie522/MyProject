#include <iostream>
#include <string>
#include <cstring>
#include <arpa/inet.h>
#include <unistd.h>
#include "nlohmann/json.hpp"
#include <opencv2/opencv.hpp>

using json = nlohmann::json;

class UdpSender{
public:
    UdpSender(const std::string& ip, uint16_t port);

    ~UdpSender();

    void sendData(const json& j);
    json packData(const std::vector<cv::Point2d>& Objects, const std::vector<cv::Vec3f>& Lanes);

private:
    json prepareData(const std::vector<cv::Point2d>& Objects, const std::vector<cv::Vec3f>& Lanes);

    bool send_json();
private:
    sockaddr_in dest_addr;
    int sock;
    json data;
};