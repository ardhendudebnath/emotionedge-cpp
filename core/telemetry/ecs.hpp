#pragma once

#include <algorithm>

#include "core/emotion/emotion_types.hpp"

namespace ee {

/// Emotion Consistency Score (blueprint 5.2): ECS = 1 - ‖VAD_src - VAD_out‖ / 2√3.
/// 1 means the synthesized voice carries exactly the source emotion; 0 means opposite corners
/// of the V·A·D cube. The success target is ECS >= 0.75.
[[nodiscard]] inline float emotion_consistency(Vad src, Vad out) {
    return std::clamp(1.0f - distance(src, out) / kVadDiameter, 0.0f, 1.0f);
}

}  // namespace ee
