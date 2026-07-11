#include "face_pipeline/face_embedder.h"

#include <cmath>
#include <stdexcept>

#include <opencv2/objdetect/face.hpp>

namespace face_pipeline {

class SFaceEmbedder::Impl {
public:
    explicit Impl(const std::string& path)
        : recognizer(cv::FaceRecognizerSF::create(path, "")) {
        if (recognizer.empty()) throw std::runtime_error("failed to load SFace model: " + path);
    }

    cv::Ptr<cv::FaceRecognizerSF> recognizer;
};

SFaceEmbedder::SFaceEmbedder(const std::string& model_path)
    : impl_(new Impl(model_path)) {
    cv::Mat validation_input(112, 112, CV_8UC3, cv::Scalar(127, 127, 127));
    extract(validation_input);
}

SFaceEmbedder::~SFaceEmbedder() = default;

std::vector<float> SFaceEmbedder::extract(const cv::Mat& aligned_face) {
    if (aligned_face.empty() || aligned_face.size() != cv::Size(112, 112) ||
        aligned_face.type() != CV_8UC3) {
        throw std::runtime_error("SFace input must be a 112x112 BGR image");
    }

    cv::Mat feature;
    impl_->recognizer->feature(aligned_face, feature);
    cv::Mat flattened = feature.reshape(1, 1);
    if (flattened.type() != CV_32F || flattened.total() != 128) {
        throw std::runtime_error("SFace output must contain 128 float values");
    }

    const float norm = static_cast<float>(cv::norm(flattened, cv::NORM_L2));
    if (!std::isfinite(norm) || norm <= 1e-12f) {
        throw std::runtime_error("SFace produced an invalid feature vector");
    }
    flattened /= norm;

    const float* begin = flattened.ptr<float>();
    std::vector<float> embedding(begin, begin + flattened.total());
    for (float value : embedding) {
        if (!std::isfinite(value)) throw std::runtime_error("SFace produced non-finite values");
    }
    return embedding;
}

std::string SFaceEmbedder::modelName() const {
    return "opencv_sface_2021dec";
}

}  // namespace face_pipeline
