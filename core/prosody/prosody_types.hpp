#pragma once

#include <array>
#include <cstddef>

namespace ee {

inline constexpr std::size_t kStyleDim = 128;             ///< StyleTTS2 style vector (blueprint 4.1)
inline constexpr std::size_t kSpeakerEmbeddingDim = 192;  ///< ECAPA-TDNN voice print (1.4)

using StyleVector = std::array<float, kStyleDim>;
using SpeakerEmbedding = std::array<float, kSpeakerEmbeddingDim>;

/// Prosody plan from the emotion controller (4.1), relative to the target language's neutral
/// baseline. The walkthrough on blueprint p.3 plans pitch +15 %, range +30 %, rate +10 %,
/// energy +4 dB, pause 80 ms and accent ×1.4 for an angry utterance.
struct ProsodyTargets {
    float pitch_pct = 0.0f;   ///< pitch-mean shift, percent
    float range_pct = 0.0f;   ///< pitch-range change, percent
    float rate_pct = 0.0f;    ///< speaking-rate change, percent
    float energy_db = 0.0f;   ///< loudness change, dB
    float pause_ms = 0.0f;    ///< pre-pause before emphasized words, ms
    float accent = 1.0f;      ///< pitch-accent multiplier on emphasized words
    float final_fall = 0.0f;  ///< final contour: +1 strongly falling ... -1 rising
    float hesitation = 1.0f;  ///< scale on hesitation (inter-word) pauses; 1 = baseline
    float tension = 0.0f;     ///< voice-quality tension, 0 (lax) ... 1 (pressed)
};

}  // namespace ee
