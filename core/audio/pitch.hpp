#pragma once

#include <span>
#include <vector>

namespace ee {

struct PitchConfig {
    int sample_rate = 16000;
    float fmin_hz = 60.0f;
    float fmax_hz = 500.0f;
    float threshold = 0.15f;       ///< YIN absolute threshold on the normalized difference
    int frame = 640;               ///< analysis frame (40 ms at 16 kHz)
    int hop = 160;                 ///< 10 ms
    float silence_dbfs = -50.0f;   ///< frames below this are unvoiced
};

struct PitchFrame {
    float f0_hz = 0.0f;        ///< 0 when unvoiced
    float periodicity = 0.0f;  ///< 1 - min normalized difference: ~1 for clean voicing
    bool voiced = false;
};

/// YIN fundamental-frequency estimator (de Cheveigné & Kawahara, 2002).
class Yin {
public:
    explicit Yin(PitchConfig config);
    /// Analyzes one frame of exactly config.frame samples.
    [[nodiscard]] PitchFrame analyze(std::span<const float> frame);
    [[nodiscard]] const PitchConfig& config() const noexcept { return cfg_; }

private:
    PitchConfig cfg_;
    int tau_min_;
    int tau_max_;
    int window_;
    std::vector<float> diff_;
};

/// Frame-by-frame pitch track of a whole signal (one frame per hop; the tail is zero-padded).
[[nodiscard]] std::vector<PitchFrame> track_pitch(std::span<const float> signal, const PitchConfig& config);

/// Semitones between two frequencies.
[[nodiscard]] float semitones(float f_hz, float ref_hz);

}  // namespace ee
