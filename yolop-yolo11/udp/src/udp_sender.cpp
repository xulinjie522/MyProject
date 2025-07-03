#include "udp_sender.h"


UdpSender::UdpSender(const std::string& ip, uint16_t port){
    sock = socket(AF_INET, SOCK_DGRAM, 0);
    if(sock < 0){
        perror("socket");
        throw std::runtime_error("Failed to create socket");
    }

    memset(&dest_addr, 0, sizeof(dest_addr));
    dest_addr.sin_family = AF_INET;
    dest_addr.sin_port = htons(port);
    if(inet_pton(AF_INET, ip.c_str(), &dest_addr.sin_addr) <= 0){
        perror("inet_pton");
        throw std::runtime_error("Invalid IP address");
    }
}

UdpSender::~UdpSender(){
    if(sock >= 0) close(sock);
}

json UdpSender::prepareData(const std::vector<cv::Point2d>& Objects, const std::vector<cv::Vec3f>& Lanes){
    json data;

    for(size_t i = 0; i < Objects.size(); ++i){
        json object;
        object["id"] = i;
        object["type"] = "object";
        object["position"] = {{"x", Objects[i].x}, {"y", Objects[i].y}};
        data["object"].emplace_back(object);
    }

    for(size_t i = 0; i < Lanes.size(); ++i){
        json lane;
        lane["id"] = i;
        lane["A"] = Lanes[i][0];
        lane["B"] = Lanes[i][1];
        lane["C"] = Lanes[i][2];
        lane["StartY"] = 0;
        lane["EndY"] = 1000;
        data["lane"].emplace_back(lane);
    }


    return data;
}

#include <iostream>
#include <string>
#include <sstream>
#include <unistd.h>
#include <arpa/inet.h>

bool UdpSender::send_json() {
    constexpr size_t MAX_UDP_SIZE = 65507;

    std::string json_str = data.dump();
    size_t total_size = json_str.size();

    // 不需要分段
    if (total_size <= MAX_UDP_SIZE) {
        ssize_t sent = sendto(sock, json_str.c_str(), json_str.size(), 0,
                              (sockaddr*)&dest_addr, sizeof(dest_addr));
        if (sent < 0) {
            perror("sendto");
            return false;
        }
        return true;
    }

    // 需要分段
    // 分段时，额外每段加一个简易头部，格式示例: "SEQ:1/3\n"
    // 所以有效内容比 MAX_UDP_SIZE 稍微少一点
    size_t payload_per_packet = MAX_UDP_SIZE - 20; // 预留头部空间

    // 计算段数
    size_t num_packets = (total_size + payload_per_packet - 1) / payload_per_packet;

    for (size_t i = 0; i < num_packets; ++i) {
        size_t start = i * payload_per_packet;
        size_t end = std::min(start + payload_per_packet, total_size);
        std::string chunk = json_str.substr(start, end - start);

        // 在每段前面加上头部信息
        // 例如: SEQ:1/5\n
        std::ostringstream oss;
        oss << "SEQ:" << (i + 1) << "/" << num_packets << "\n" << chunk;
        std::string packet_data = oss.str();

        ssize_t sent = sendto(sock, packet_data.c_str(), packet_data.size(), 0,
                              (sockaddr*)&dest_addr, sizeof(dest_addr));
        if (sent < 0) {
            perror("sendto");
            return false;
        }
    }

    return true;
}

json UdpSender::packData(const std::vector<cv::Point2d>& Objects, const std::vector<cv::Vec3f>& Lanes){
    return prepareData(Objects, Lanes);
}

void UdpSender::sendData(const json& j){
    data = j;
    if(!send_json()){
        std::cerr << "Failed to send!" << std::endl;
    }
}