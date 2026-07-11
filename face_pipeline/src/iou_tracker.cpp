#include "face_pipeline/iou_tracker.h"

#include <algorithm>

namespace face_pipeline {

IouTracker::IouTracker(const IouTrackerConfig& config) : config_(config) {}

float IouTracker::intersectionOverUnion(const cv::Rect& a, const cv::Rect& b) {
    const cv::Rect intersection = a & b;
    if (intersection.area() <= 0) return 0.0f;
    const float union_area = static_cast<float>(a.area() + b.area() - intersection.area());
    return union_area > 0.0f ? static_cast<float>(intersection.area()) / union_area : 0.0f;
}

void IouTracker::expire(SteadyTime now) {
    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(), [&](const Track& track) {
        return now - track.last_seen >= config_.track_ttl;
    }), tracks_.end());
}

std::vector<TrackedFace> IouTracker::update(
    const std::vector<FaceObservation>& observations, SteadyTime now) {
    expire(now);

    struct Candidate {
        std::size_t track;
        std::size_t observation;
        float iou;
    };
    std::vector<Candidate> candidates;
    for (std::size_t track = 0; track < tracks_.size(); ++track) {
        for (std::size_t observation = 0; observation < observations.size(); ++observation) {
            const float iou = intersectionOverUnion(tracks_[track].observation.bbox,
                                                    observations[observation].bbox);
            if (iou >= config_.match_threshold) candidates.push_back({track, observation, iou});
        }
    }
    std::sort(candidates.begin(), candidates.end(), [](const Candidate& left,
                                                        const Candidate& right) {
        return left.iou > right.iou;
    });

    std::vector<bool> track_used(tracks_.size(), false);
    std::vector<bool> observation_used(observations.size(), false);
    std::vector<TrackedFace> result;
    for (const Candidate& candidate : candidates) {
        if (track_used[candidate.track] || observation_used[candidate.observation]) continue;
        Track& track = tracks_[candidate.track];
        track.observation = observations[candidate.observation];
        track.last_seen = now;
        track_used[candidate.track] = true;
        observation_used[candidate.observation] = true;
        TrackedFace tracked_face;
        tracked_face.track_id = track.id;
        tracked_face.observation = track.observation;
        tracked_face.is_new = false;
        result.push_back(tracked_face);
    }

    for (std::size_t observation = 0; observation < observations.size(); ++observation) {
        if (observation_used[observation]) continue;
        Track track{next_track_id_++, observations[observation], now};
        tracks_.push_back(track);
        TrackedFace tracked_face;
        tracked_face.track_id = track.id;
        tracked_face.observation = track.observation;
        tracked_face.is_new = true;
        result.push_back(tracked_face);
    }
    return result;
}

bool IouTracker::hasActiveTracks(SteadyTime now) const {
    for (const Track& track : tracks_) {
        if (now - track.last_seen < config_.track_ttl) return true;
    }
    return false;
}

std::vector<std::uint64_t> IouTracker::activeTrackIds() const {
    std::vector<std::uint64_t> ids;
    ids.reserve(tracks_.size());
    for (const Track& track : tracks_) ids.push_back(track.id);
    return ids;
}

std::size_t IouTracker::size() const {
    return tracks_.size();
}

}  // namespace face_pipeline
