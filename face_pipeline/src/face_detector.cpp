#include "face_pipeline/face_detector.h"

#include <algorithm>

#include "facedetectcnn.h"

namespace face_pipeline {

FaceDetector::FaceDetector(float minimum_score)
    : minimum_score_(minimum_score), result_buffer_(FACEDETECTION_RESULT_BUFFER_SIZE) {}

std::vector<FaceObservation> FaceDetector::detect(const cv::Mat& frame) {
    std::vector<FaceObservation> observations;
    if (frame.empty() || frame.type() != CV_8UC3) return observations;

    int* results = facedetect_cnn(result_buffer_.data(), frame.data, frame.cols, frame.rows,
                                  static_cast<int>(frame.step));
    if (!results) return observations;

    observations.reserve(static_cast<std::size_t>(std::max(0, results[0])));
    for (int i = 0; i < results[0]; ++i) {
        short* values = reinterpret_cast<short*>(results + 1) +
                        FACEDETECTION_RESULT_STRIDE_SHORTS * i;
        const float score = static_cast<float>(values[0]) / 100.0f;
        cv::Rect bbox(values[1], values[2], values[3], values[4]);
        bbox &= cv::Rect(0, 0, frame.cols, frame.rows);
        if (score <= minimum_score_ || bbox.width <= 0 || bbox.height <= 0) continue;

        FaceObservation observation;
        observation.bbox = bbox;
        observation.score = score;
        for (std::size_t landmark = 0; landmark < observation.landmarks.size(); ++landmark) {
            observation.landmarks[landmark] = cv::Point2f(
                static_cast<float>(values[5 + landmark * 2]),
                static_cast<float>(values[6 + landmark * 2]));
        }
        observations.push_back(observation);
    }
    return observations;
}

}  // namespace face_pipeline
