#include "face_pipeline/motion_detector.h"

#include <opencv2/imgproc.hpp>

namespace face_pipeline {

MotionDetector::MotionDetector(const MotionDetectorConfig& config) : config_(config) {}

bool MotionDetector::shouldProcess(const cv::Mat& frame, SteadyTime now,
                                   bool has_active_tracks) {
    if (frame.empty()) return false;

    cv::Mat resized;
    cv::Mat gray;
    cv::resize(frame, resized, config_.analysis_size);
    cv::cvtColor(resized, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0.0);

    if (!initialized_) {
        gray.copyTo(previous_gray_);
        last_motion_ = now - config_.active_hold;
        last_request_ = now;
        initialized_ = true;
        return false;
    }

    cv::Mat difference;
    cv::Mat changed;
    cv::absdiff(previous_gray_, gray, difference);
    cv::threshold(difference, changed, config_.pixel_threshold, 255, cv::THRESH_BINARY);
    cv::morphologyEx(changed, changed, cv::MORPH_OPEN,
                     cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
    gray.copyTo(previous_gray_);

    const double changed_ratio = static_cast<double>(cv::countNonZero(changed)) /
                                 static_cast<double>(changed.total());
    if (changed_ratio >= config_.changed_ratio_threshold) last_motion_ = now;

    const std::chrono::milliseconds scan_interval = has_active_tracks
        ? config_.tracked_scan_interval : config_.idle_scan_interval;
    const bool motion_active = now - last_motion_ < config_.active_hold;
    const bool scan_due = now - last_request_ >= scan_interval;
    if (!motion_active && !scan_due) return false;

    last_request_ = now;
    return true;
}

void MotionDetector::reset() {
    previous_gray_.release();
    initialized_ = false;
}

}  // namespace face_pipeline
