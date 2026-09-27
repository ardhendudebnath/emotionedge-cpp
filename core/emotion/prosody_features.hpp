#pragma once

#include <span>
#include <vector>

#include "core/audio/pitch.hpp"

namespace ee {

/// One 10 ms analysis frame (blueprint 2.2 "PROSODY: F0 · energy · rate · jitter").
struct ProsodyFrame {
    float f0_hz = 0.0f;  ///< 0 when unvoiced
    bool voiced = false;
    float energy_db = -120.0f;
    float hf_db = 0.0f;  ///< spectral-tilt proxy (see hf_ratio_db)
};

/// Incremental prosody analysis. Frame i is centered at t = i · hop from the first sample
/// pushed, so its timeline matches the ASR word timestamps of the same utterance.
class ProsodyTracker {
public:
    explicit ProsodyTracker(int sample_rate = 16000);
    void reset();
    void push(std::span<const float> audio);
    /// Analyzes what is left at end of utterance (zero-padded).
    void flush();
    [[nodiscard]] const std::vector<ProsodyFrame>& frames() const noexcept { return frames_; }
    [[nodiscard]] float hop_seconds() const noexcept { return static_cast<float>(hop_) / static_cast<float>(rate_); }

private:
    void analyze_ready(bool flushing);
    void analyze_one(std::span<const float> frame);

    int rate_;
    std::size_t frame_;
    std::size_t hop_;
    Yin yin_;
    std::vector<float> buffer_;   // samples from absolute index consumed_ (zero prefix included)
    std::vector<float> scratch_;
    std::size_t consumed_ = 0;
    std::size_t pushed_ = 0;
    std::vector<ProsodyFrame> frames_;
};

/// Utterance- or window-level prosody statistics.
struct ProsodySummary {
    float duration_s = 0.0f;         ///< from the first to the last active frame
    int active_frames = 0;
    int voiced_frames = 0;
    float voiced_ratio = 0.0f;
    float f0_median_hz = 0.0f;
    float f0_range_st = 0.0f;        ///< p90 - p10, semitones
    float energy_db = -120.0f;       ///< mean over active frames
    float energy_peak_db = -120.0f;
    float hf_db = 0.0f;              ///< mean spectral-tilt proxy over voiced frames
    float syllable_rate = 0.0f;      ///< energy-envelope peaks per second
    float pause_ratio = 0.0f;        ///< inactive frames between the first and last active ones
    float final_slope_st_s = 0.0f;   ///< F0 slope over the last 300 ms of voicing, semitones/s
    float jitter = 0.0f;             ///< mean |ΔF0| / F0 between consecutive voiced frames
};

[[nodiscard]] ProsodySummary summarize(std::span<const ProsodyFrame> frames, float hop_s);

}  // namespace ee
