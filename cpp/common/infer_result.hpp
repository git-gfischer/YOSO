#pragma once

#include <vector>

template<typename DetectionT>
struct InferResult {
    std::vector<DetectionT> detections;
    double preprocess_ms = 0.0;
    double infer_ms      = 0.0;
    double postprocess_ms = 0.0;
    double total_ms      = 0.0;

    double fps() const {
        return total_ms > 0.0 ? 1000.0 / total_ms : 0.0;
    }
};
