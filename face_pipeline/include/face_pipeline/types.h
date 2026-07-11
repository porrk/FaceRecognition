#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace face_pipeline {

using SteadyTime = std::chrono::steady_clock::time_point;

struct FramePacket {
    cv::Mat frame;
    std::uint64_t frame_id = 0;
    SteadyTime captured_at;
    std::int64_t captured_at_unix_ms = 0;
};

struct FaceObservation {
    cv::Rect bbox;
    float score = 0.0f;
    std::array<cv::Point2f, 5> landmarks;
};

struct TrackedFace {
    std::uint64_t track_id = 0;
    FaceObservation observation;
    bool is_new = false;
};

struct FaceFeatureEvent {
    std::uint64_t track_id = 0;
    std::uint64_t frame_id = 0;
    std::int64_t captured_at_unix_ms = 0;
    FaceObservation observation;
    std::string embedding_model;
    std::vector<float> embedding;
};

}  // namespace face_pipeline
