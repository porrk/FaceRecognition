#include "face_pipeline/face_aligner.h"

#include <cmath>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace face_pipeline {
namespace {

const std::vector<cv::Point2f> kSFaceTemplate = {
    {38.2946f, 51.6963f}, {73.5318f, 51.5014f}, {56.0252f, 71.7366f},
    {41.5493f, 92.3655f}, {70.7299f, 92.2041f}
};

}  // namespace

cv::Size FaceAligner::outputSize() {
    return cv::Size(112, 112);
}

bool FaceAligner::align(const cv::Mat& frame, const FaceObservation& face,
                        cv::Mat& aligned) const {
    if (frame.empty() || face.bbox.width < 12 || face.bbox.height < 12) return false;

    std::vector<cv::Point2f> source(face.landmarks.begin(), face.landmarks.end());
    for (const cv::Point2f& point : source) {
        if (!std::isfinite(point.x) || !std::isfinite(point.y)) return false;
    }

    cv::Mat inliers;
    cv::Mat transform = cv::estimateAffinePartial2D(source, kSFaceTemplate, inliers, cv::LMEDS);
    if (transform.empty() || transform.rows != 2 || transform.cols != 3) return false;

    cv::warpAffine(frame, aligned, transform, outputSize(), cv::INTER_LINEAR,
                   cv::BORDER_CONSTANT, cv::Scalar());
    return !aligned.empty();
}

}  // namespace face_pipeline
