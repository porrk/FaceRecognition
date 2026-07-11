#include "face_pipeline/face_pipeline.h"

#include <iostream>
#include <stdexcept>

namespace face_pipeline {

FacePipeline::FacePipeline(std::unique_ptr<FaceEmbedder> embedder,
                           std::unique_ptr<ResultSink> sink)
    : embedder_(std::move(embedder)), sink_(std::move(sink)) {
    if (!embedder_ || !sink_) throw std::invalid_argument("pipeline dependencies are required");
}

FacePipeline::~FacePipeline() {
    stop();
}

void FacePipeline::start() {
    std::lock_guard<std::mutex> lock(mailbox_mutex_);
    if (running_) return;
    running_ = true;
    worker_ = std::thread(&FacePipeline::workerLoop, this);
}

void FacePipeline::stop() {
    {
        std::lock_guard<std::mutex> lock(mailbox_mutex_);
        if (!running_) return;
        running_ = false;
        has_mail_ = false;
    }
    mailbox_cv_.notify_all();
    if (worker_.joinable()) worker_.join();
}

bool FacePipeline::submit(const FramePacket& packet) {
    if (!motion_detector_.shouldProcess(packet.frame, packet.captured_at,
                                        has_active_tracks_.load())) return false;

    std::lock_guard<std::mutex> lock(mailbox_mutex_);
    if (!running_) return false;
    submitted_frames_.fetch_add(1);
    if (has_mail_) overwritten_frames_.fetch_add(1);
    mailbox_ = packet;
    mailbox_.frame = packet.frame.clone();
    has_mail_ = true;
    mailbox_cv_.notify_one();
    return true;
}

std::vector<TrackedFace> FacePipeline::latestTracks() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return latest_tracks_;
}

void FacePipeline::workerLoop() {
    while (true) {
        FramePacket packet;
        {
            std::unique_lock<std::mutex> lock(mailbox_mutex_);
            mailbox_cv_.wait(lock, [&] { return has_mail_ || !running_; });
            if (!running_) break;
            packet = mailbox_;
            has_mail_ = false;
        }
        process(packet);
    }
}

void FacePipeline::process(const FramePacket& packet) {
    try {
        ++processed_frames_;
        const std::vector<FaceObservation> observations = face_detector_.detect(packet.frame);
        const std::vector<TrackedFace> tracked = tracker_.update(observations, packet.captured_at);
        has_active_tracks_.store(tracker_.hasActiveTracks(packet.captured_at));

        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            latest_tracks_ = tracked;
            const std::vector<std::uint64_t> active_ids = tracker_.activeTrackIds();
            const std::set<std::uint64_t> active(active_ids.begin(), active_ids.end());
            for (std::set<std::uint64_t>::iterator it = published_tracks_.begin();
                 it != published_tracks_.end();) {
                if (active.count(*it) == 0) {
                    it = published_tracks_.erase(it);
                } else {
                    ++it;
                }
            }
        }

        for (const TrackedFace& face : tracked) {
            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                if (published_tracks_.count(face.track_id) != 0) continue;
            }

            try {
                cv::Mat aligned;
                if (!face_aligner_.align(packet.frame, face.observation, aligned)) {
                    ++alignment_failures_;
                    continue;
                }

                FaceFeatureEvent event;
                event.track_id = face.track_id;
                event.frame_id = packet.frame_id;
                event.captured_at_unix_ms = packet.captured_at_unix_ms;
                event.observation = face.observation;
                event.embedding_model = embedder_->modelName();
                event.embedding = embedder_->extract(aligned);
                if (sink_->publish(event)) {
                    std::lock_guard<std::mutex> lock(state_mutex_);
                    published_tracks_.insert(face.track_id);
                } else {
                    ++publish_failures_;
                }
            } catch (const cv::Exception& error) {
                ++embedding_failures_;
                std::cerr << "feature extraction failed for track " << face.track_id
                          << ": " << error.what() << std::endl;
            } catch (const std::exception& error) {
                ++embedding_failures_;
                std::cerr << "feature extraction failed for track " << face.track_id
                          << ": " << error.what() << std::endl;
            }
        }

        if (processed_frames_ % 100 == 0) {
            std::clog << "face_pipeline_stats submitted=" << submitted_frames_.load()
                      << " overwritten=" << overwritten_frames_.load()
                      << " processed=" << processed_frames_
                      << " align_failed=" << alignment_failures_
                      << " embed_failed=" << embedding_failures_
                      << " publish_failed=" << publish_failures_ << std::endl;
        }
    } catch (const cv::Exception& error) {
        std::cerr << "face pipeline OpenCV error: " << error.what() << std::endl;
    } catch (const std::exception& error) {
        std::cerr << "face pipeline error: " << error.what() << std::endl;
    }
}

}  // namespace face_pipeline
