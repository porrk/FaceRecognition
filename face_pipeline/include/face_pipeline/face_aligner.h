#pragma once

#include <opencv2/core.hpp>

#include "face_pipeline/types.h"

namespace face_pipeline {

class FaceAligner {
public:
    static cv::Size outputSize();
    bool align(const cv::Mat& frame, const FaceObservation& face, cv::Mat& aligned) const;
};

}  // namespace face_pipeline
