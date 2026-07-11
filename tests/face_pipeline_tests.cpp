#include <chrono>
#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "face_pipeline/face_aligner.h"
#include "face_pipeline/face_embedder.h"
#include "face_pipeline/iou_tracker.h"
#include "face_pipeline/motion_detector.h"
#include "face_pipeline/result_sink.h"

namespace {

using face_pipeline::SteadyTime;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

face_pipeline::FaceObservation observation(int x, int y, int width, int height) {
    face_pipeline::FaceObservation face;
    face.bbox = cv::Rect(x, y, width, height);
    face.score = 0.9f;
    return face;
}

void testMotionDetector() {
    face_pipeline::MotionDetector detector;
    const SteadyTime start = std::chrono::steady_clock::now();
    const cv::Mat black(180, 320, CV_8UC3, cv::Scalar::all(0));
    cv::Mat moving = black.clone();
    cv::rectangle(moving, cv::Rect(40, 40, 80, 80), cv::Scalar::all(255), -1);

    require(!detector.shouldProcess(black, start, false), "first frame must initialize");
    require(!detector.shouldProcess(black, start + std::chrono::milliseconds(100), false),
            "static frame must not trigger");
    require(detector.shouldProcess(moving, start + std::chrono::milliseconds(200), false),
            "local movement must trigger");
    require(detector.shouldProcess(moving, start + std::chrono::milliseconds(600), false),
            "motion hold must remain active");
    require(detector.shouldProcess(moving, start + std::chrono::milliseconds(1600), true),
            "active tracks must force a one-second scan");
}

void testFaceAligner() {
    face_pipeline::FaceObservation face = observation(20, 20, 80, 80);
    face.landmarks = {{{38.2946f, 51.6963f}, {73.5318f, 51.5014f},
                       {56.0252f, 71.7366f}, {41.5493f, 92.3655f},
                       {70.7299f, 92.2041f}}};
    cv::Mat input(112, 112, CV_8UC3);
    for (int row = 0; row < input.rows; ++row) {
        for (int col = 0; col < input.cols; ++col) {
            input.at<cv::Vec3b>(row, col) = cv::Vec3b(row, col, (row + col) / 2);
        }
    }

    face_pipeline::FaceAligner aligner;
    cv::Mat aligned;
    require(aligner.align(input, face, aligned), "template landmarks must align");
    require(aligned.size() == cv::Size(112, 112), "aligned size must be 112x112");
    require(cv::norm(input, aligned, cv::NORM_INF) <= 2.0, "template alignment must be identity");

    face.bbox = cv::Rect(0, 0, 5, 5);
    require(!aligner.align(input, face, aligned), "tiny faces must be rejected");
}

void testIouTracker() {
    face_pipeline::IouTracker tracker;
    const SteadyTime start = std::chrono::steady_clock::now();

    std::vector<face_pipeline::TrackedFace> first = tracker.update(
        {observation(10, 10, 50, 50), observation(200, 10, 50, 50)}, start);
    require(first.size() == 2 && first[0].is_new && first[1].is_new,
            "first observations must create tracks");
    const std::uint64_t first_id = first[0].track_id;

    std::vector<face_pipeline::TrackedFace> matched = tracker.update(
        {observation(14, 12, 50, 50)}, start + std::chrono::milliseconds(500));
    require(matched.size() == 1 && matched[0].track_id == first_id && !matched[0].is_new,
            "overlapping observation must preserve track id");

    std::vector<face_pipeline::TrackedFace> expired = tracker.update(
        {observation(14, 12, 50, 50)}, start + std::chrono::milliseconds(2600));
    require(expired.size() == 1 && expired[0].track_id != first_id && expired[0].is_new,
            "expired observation must receive a new track id");
}

void testLoggingSink() {
    std::ostringstream output;
    face_pipeline::LoggingResultSink sink(output);
    face_pipeline::FaceFeatureEvent event;
    event.track_id = 7;
    event.frame_id = 42;
    event.embedding_model = "test_model";
    event.embedding.assign(128, 0.0f);
    require(sink.publish(event), "logging sink must report success");
    require(output.str().find("track_id=7") != std::string::npos,
            "logging sink must include the track id");
    require(output.str().find("embedding_dim=128") != std::string::npos,
            "logging sink must include embedding dimension");
}

void testSFaceModel() {
    face_pipeline::SFaceEmbedder embedder(
        std::string(SOURCE_ROOT) + "/models/face_recognition_sface_2021dec.onnx");
    cv::Mat input(112, 112, CV_8UC3, cv::Scalar(90, 120, 150));
    const std::vector<float> embedding = embedder.extract(input);
    require(embedding.size() == 128, "SFace must return 128 values");
    double squared_norm = 0.0;
    for (float value : embedding) {
        require(std::isfinite(value), "SFace values must be finite");
        squared_norm += static_cast<double>(value) * value;
    }
    require(std::abs(std::sqrt(squared_norm) - 1.0) < 1e-5,
            "SFace embedding must be L2 normalized");
}

}  // namespace

int main() {
    try {
        testMotionDetector();
        testFaceAligner();
        testIouTracker();
        testLoggingSink();
        testSFaceModel();
    } catch (const std::exception& error) {
        std::cerr << "face_pipeline_tests failed: " << error.what() << std::endl;
        return 1;
    }
    std::cout << "face_pipeline_tests passed" << std::endl;
    return 0;
}
