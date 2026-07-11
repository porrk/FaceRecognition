/*
 * @Author: Liu zhenyu
 * @Description: 网络摄像头接收器（重构版）
 *               仅负责从视频流抓取最新帧，不做任何检测/显示。
 *               主程序通过 getLatestFrame() 拉取最新帧后自行处理。
 */

#ifndef NETWORK_CAMERA_RECEIVER_H
#define NETWORK_CAMERA_RECEIVER_H

#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <mutex>
#include <atomic>

class NetworkCameraReceiver {
public:
    NetworkCameraReceiver(const std::string& host_ip, int port = 5000);
    ~NetworkCameraReceiver();

    // 打开视频流
    bool connect();
    // 启动后台抓帧线程（不阻塞）
    void start();
    // 停止抓帧线程并 join
    void stop();
    // 取最新帧（深拷贝到 out）；自上次取帧后若无新帧则返回 false
    bool getLatestFrame(cv::Mat& out);
    // 视频流是否已打开
    bool isOpened() const;

private:
    // 后台抓帧循环
    void captureLoop();

    std::string stream_url;
    cv::VideoCapture cap;

    std::thread capture_thread;
    std::mutex mtx;                 // 保护 latest_frame
    cv::Mat latest_frame;           // 最新一帧
    std::atomic<bool> has_new_frame{false};
    std::atomic<bool> running{false};
};

#endif // NETWORK_CAMERA_RECEIVER_H
