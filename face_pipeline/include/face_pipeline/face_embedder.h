#pragma once

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace face_pipeline {

class FaceEmbedder {
public:
    virtual ~FaceEmbedder() = default;
    virtual std::vector<float> extract(const cv::Mat& aligned_face) = 0;
    virtual std::string modelName() const = 0;
};

class SFaceEmbedder : public FaceEmbedder {
public:
    explicit SFaceEmbedder(const std::string& model_path);
    ~SFaceEmbedder() override;
    std::vector<float> extract(const cv::Mat& aligned_face) override;
    std::string modelName() const override;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace face_pipeline
