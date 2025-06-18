#include <iostream>
#include <string>
#include <cstring>
#include <arpa/inet.h>
#include <unistd.h>
#include "nlohmann/json.hpp"

using json = nlohmann::json;

class UdpSender{
public:
    UdpSender(const std::string& ip, uint16_t port);

    ~UdpSender();

    bool send_json(const json& j);
private:
    sockaddr_in dest_addr;
    int sock;
};