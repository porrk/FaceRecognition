/*
 * @Author: Liu zhenyu
 * @Description: 主程序（重构版）
 *               流程：1. 从 network_camera_receiver 取最新帧；
 *                     2. 调用 libfacedetection 进行人脸检测；
 *                     3. 绘制并显示检测结果。
 *               libfacedetection 以独立动态库形式链接。
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

#include "network_camera_receiver.h"
#include "facedetectcnn.h"

// 将 libfacedetection 的检测结果绘制到帧上
// pResults[0] = 人脸数量；其后每张脸占 FACEDETECTION_RESULT_STRIDE_SHORTS 个 short
static void drawFaces(cv::Mat& frame, const int* pResults) {
    if (!pResults) return;
    int faces = pResults[0];

    for (int i = 0; i < faces; i++) {
        // 每个人脸结果占用 FACEDETECTION_RESULT_STRIDE_SHORTS(=16) 个 short：
        //   p[0]=置信度(score*100), p[1..4]=x/y/w/h,
        //   p[5..14]=5个关键点(x,y交替), p[15]=对齐填充
        short* p = ((short*)(pResults + 1)) + FACEDETECTION_RESULT_STRIDE_SHORTS * i;

        int confidence = p[0];
        int x = p[1];
        int y = p[2];
        int w = p[3];
        int h = p[4];

        // 只显示置信度大于 60 的结果以过滤误检
        if (confidence > 60) {
            cv::rectangle(frame, cv::Rect(x, y, w, h), cv::Scalar(0, 255, 0), 2);

            // 置信度标注
            cv::putText(frame, std::to_string(confidence),
                        cv::Point(x, y - 5), cv::FONT_HERSHEY_SIMPLEX,
                        0.5, cv::Scalar(0, 255, 0), 1);

            // 五个特征点
            cv::circle(frame, cv::Point(p[5],  p[6]),  2, cv::Scalar(255, 0, 0),   -1);
            cv::circle(frame, cv::Point(p[7],  p[8]),  2, cv::Scalar(0, 0, 255),   -1);
            cv::circle(frame, cv::Point(p[9],  p[10]), 2, cv::Scalar(0, 255, 0),   -1);
            cv::circle(frame, cv::Point(p[11], p[12]), 2, cv::Scalar(255, 0, 255), -1);
            cv::circle(frame, cv::Point(p[13], p[14]), 2, cv::Scalar(0, 255, 255), -1);
        }
    }
}

int main() {
    // 替换为你查到的视频流宿主机 IP
    std::string host_ip = "127.0.0.1";

    // 1) 接收端：仅负责抓帧
    NetworkCameraReceiver receiver(host_ip, 5000);
    if (!receiver.connect()) {
        std::cerr << "连接视频流失败，退出。" << std::endl;
        return -1;
    }
    receiver.start();  // 启动后台抓帧线程

    // libfacedetection 结果缓冲区（大小由库定义的宏决定）
    unsigned char* pBuffer = new unsigned char[FACEDETECTION_RESULT_BUFFER_SIZE];

    const std::string window_name = "Face Detection";
    cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);

    std::cout << "人脸检测已启动，按 ESC 退出..." << std::endl;

    cv::Mat frame;
    while (true) {
        // ---- 步骤 1：接收最新帧 ----
        if (receiver.getLatestFrame(frame) && !frame.empty()) {
            // ---- 步骤 2：调用 libfacedetection 检测 ----
            // 注意：OpenCV 采集到的帧为 BGR，正好符合 facedetect_cnn 的输入要求
            int* pResults = facedetect_cnn(pBuffer,
                                           frame.data,
                                           frame.cols,
                                           frame.rows,
                                           (int)frame.step);

            // ---- 步骤 3：显示检测结果 ----
            drawFaces(frame, pResults);
            cv::imshow(window_name, frame);
        }

        if (cv::waitKey(30) == 27) break;  // ESC 退出
    }

    receiver.stop();
    delete[] pBuffer;
    cv::destroyAllWindows();
    return 0;
}
