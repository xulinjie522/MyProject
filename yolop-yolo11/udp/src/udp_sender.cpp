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

bool UdpSender::send_json(const json& j){
    std::string json_str = j.dump();
    ssize_t sent = sendto(sock, json_str.c_str(), json_str.size(), 0,
                            (sockaddr*)&dest_addr, sizeof(dest_addr));
    if(sent < 0){
        perror("sendto");
        return false;
    }
    std::cout << "Sent Json: " << json_str << std::endl;
    return true;
}