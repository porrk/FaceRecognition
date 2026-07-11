#pragma once

#include <chrono>

#include <opencv2/core.hpp>

#include "face_pipeline/types.h"

namespace face_pipeline {

struct MotionDetectorConfig {
    cv::Size analysis_size{320, 180};
    int pixel_threshold = 20;
    double changed_ratio_threshold = 0.01;
    std::chrono::milliseconds active_hold{500};
    std::chrono::milliseconds tracked_scan_interval{1000};
    std::chrono::milliseconds idle_scan_interval{5000};
};

class MotionDetector {
public:
    explicit MotionDetector(const MotionDetectorConfig& config = MotionDetectorConfig());
    bool shouldProcess(const cv::Mat& frame, SteadyTime now, bool has_active_tracks);
    void reset();

private:
    MotionDetectorConfig config_;
    cv::Mat previous_gray_;
    SteadyTime last_motion_;
    SteadyTime last_request_;
    bool initialized_ = false;
};

}  // namespace face_pipeline
