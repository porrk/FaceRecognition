#include "face_pipeline/result_sink.h"

namespace face_pipeline {

LoggingResultSink::LoggingResultSink(std::ostream& output) : output_(output) {}

bool LoggingResultSink::publish(const FaceFeatureEvent& event) {
    output_ << "face_feature track_id=" << event.track_id
            << " frame_id=" << event.frame_id
            << " captured_at_ms=" << event.captured_at_unix_ms
            << " bbox=" << event.observation.bbox.x << ',' << event.observation.bbox.y
            << ',' << event.observation.bbox.width << ',' << event.observation.bbox.height
            << " score=" << event.observation.score
            << " model=" << event.embedding_model
            << " embedding_dim=" << event.embedding.size() << std::endl;
    return static_cast<bool>(output_);
}

}  // namespace face_pipeline
