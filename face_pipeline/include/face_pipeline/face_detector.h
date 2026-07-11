#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include "face_pipeline/types.h"

namespace face_pipeline {

class FaceDetector {
public:
    explicit FaceDetector(float minimum_score = 0.60f);
    std::vector<FaceObservation> detect(const cv::Mat& frame);

private:
    float minimum_score_;
    std::vector<unsigned char> result_buffer_;
};

}  // namespace face_pipeline
