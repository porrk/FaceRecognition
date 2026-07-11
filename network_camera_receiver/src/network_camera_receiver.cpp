/*
 * @Author: Liu zhenyu
 * @Description: 网络摄像头接收器（重构版）实现
 *               后台线程持续抓帧并更新最新帧缓冲；主线程按需拉取。
 */

#include "network_camera_receiver.h"
#include <iostream>
#include <chrono>

NetworkCameraReceiver::NetworkCameraReceiver(const std::string& host_ip, int port) {
    stream_url = "http://" + host_ip + ":" + std::to_string(port) + "/video";
}

NetworkCameraReceiver::~NetworkCameraReceiver() {
    stop();
    if (cap.isOpened()) cap.release();
}

bool NetworkCameraReceiver::connect() {
    std::cout << "尝试连接到视频流: " << stream_url << std::endl;
    cap.open(stream_url);
    if (!cap.isOpened()) {
        std::cerr << "无法打开视频流: " << stream_url << std::endl;
        return false;
    }
    std::cout << "视频流连接成功。" << std::endl;
    return true;
}

bool NetworkCameraReceiver::isOpened() const {
    return cap.isOpened();
}

void NetworkCameraReceiver::start() {
    // 已在运行则直接返回
    if (running.exchange(true)) return;
    capture_thread = std::thread(&NetworkCameraReceiver::captureLoop, this);
}

void NetworkCameraReceiver::stop() {
    // 未运行则直接返回
    if (!running.exchange(false)) return;
    if (capture_thread.joinable()) capture_thread.join();
}

void NetworkCameraReceiver::captureLoop() {
    cv::Mat frame;
    while (running.load()) {
        cap >> frame;
        if (frame.empty()) {
            // 流暂未就绪或读取失败，稍作等待避免空转占满 CPU
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        {
            std::lock_guard<std::mutex> lock(mtx);
            // 仅保留最新一帧，旧帧直接丢弃
            frame.copyTo(latest_frame);
        }
        has_new_frame.store(true);
    }
}

bool NetworkCameraReceiver::getLatestFrame(cv::Mat& out) {
    if (!has_new_frame.load()) return false;
    // 先清标记：若在此期间后台又写入新帧，标记会被再次置 true，下一轮可取到
    has_new_frame.store(false);
    std::lock_guard<std::mutex> lock(mtx);
    if (latest_frame.empty()) return false;
    latest_frame.copyTo(out);
    return true;
}
