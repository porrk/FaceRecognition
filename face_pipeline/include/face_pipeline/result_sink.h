#pragma once

#include <ostream>

#include "face_pipeline/types.h"

namespace face_pipeline {

class ResultSink {
public:
    virtual ~ResultSink() = default;
    virtual bool publish(const FaceFeatureEvent& event) = 0;
};

class LoggingResultSink : public ResultSink {
public:
    explicit LoggingResultSink(std::ostream& output);
    bool publish(const FaceFeatureEvent& event) override;

private:
    std::ostream& output_;
};

}  // namespace face_pipeline
