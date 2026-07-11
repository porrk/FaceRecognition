#pragma once

#include <chrono>
#include <cstdint>
#include <vector>

#include "face_pipeline/types.h"

namespace face_pipeline {

struct IouTrackerConfig {
    float match_threshold = 0.30f;
    std::chrono::milliseconds track_ttl{2000};
};

class IouTracker {
public:
    explicit IouTracker(const IouTrackerConfig& config = IouTrackerConfig());
    std::vector<TrackedFace> update(const std::vector<FaceObservation>& observations,
                                    SteadyTime now);
    bool hasActiveTracks(SteadyTime now) const;
    std::vector<std::uint64_t> activeTrackIds() const;
    std::size_t size() const;

private:
    struct Track {
        std::uint64_t id;
        FaceObservation observation;
        SteadyTime last_seen;
    };

    static float intersectionOverUnion(const cv::Rect& a, const cv::Rect& b);
    void expire(SteadyTime now);

    IouTrackerConfig config_;
    std::vector<Track> tracks_;
    std::uint64_t next_track_id_ = 1;
};

}  // namespace face_pipeline
