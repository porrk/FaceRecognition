#pragma once

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

#include "face_pipeline/face_aligner.h"
#include "face_pipeline/face_detector.h"
#include "face_pipeline/face_embedder.h"
#include "face_pipeline/iou_tracker.h"
#include "face_pipeline/motion_detector.h"
#include "face_pipeline/result_sink.h"
#include "face_pipeline/types.h"

namespace face_pipeline {

class FacePipeline {
public:
    FacePipeline(std::unique_ptr<FaceEmbedder> embedder,
                 std::unique_ptr<ResultSink> sink);
    ~FacePipeline();

    void start();
    void stop();
    bool submit(const FramePacket& packet);
    std::vector<TrackedFace> latestTracks() const;

private:
    void workerLoop();
    void process(const FramePacket& packet);

    MotionDetector motion_detector_;
    FaceDetector face_detector_;
    FaceAligner face_aligner_;
    IouTracker tracker_;
    std::unique_ptr<FaceEmbedder> embedder_;
    std::unique_ptr<ResultSink> sink_;

    mutable std::mutex state_mutex_;
    std::vector<TrackedFace> latest_tracks_;
    std::set<std::uint64_t> published_tracks_;

    std::mutex mailbox_mutex_;
    std::condition_variable mailbox_cv_;
    FramePacket mailbox_;
    bool has_mail_ = false;
    bool running_ = false;
    std::atomic<bool> has_active_tracks_{false};
    std::atomic<std::uint64_t> submitted_frames_{0};
    std::atomic<std::uint64_t> overwritten_frames_{0};
    std::uint64_t processed_frames_ = 0;
    std::uint64_t alignment_failures_ = 0;
    std::uint64_t embedding_failures_ = 0;
    std::uint64_t publish_failures_ = 0;
    std::thread worker_;
};

}  // namespace face_pipeline
