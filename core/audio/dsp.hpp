#pragma once

#include <algorithm>
#include <cmath>
#include <memory>
#include <span>
#include <string_view>

namespace ee {

[[nodiscard]] inline float db_to_gain(float db) { return std::pow(10.0f, db / 20.0f); }
[[nodiscard]] inline float gain_to_db(float gain) { return 20.0f * std::log10(std::max(gain, 1e-9f)); }

/// RMS level in dBFS (a full-scale sine is -3 dBFS); -120 for silence.
[[nodiscard]] float rms_dbfs(std::span<const float> x);
[[nodiscard]] float peak_abs(std::span<const float> x);

/// Spectral-tilt proxy: energy of the first difference (x[n] - 0.95·x[n-1]) relative to the
/// signal energy, in dB. Level-independent and higher for pressed/loud voices, whose spectra
/// are flatter: one of the arousal cues the emotion engine uses.
[[nodiscard]] float hf_ratio_db(std::span<const float> x);

/// Slow automatic gain control with a soft limiter (blueprint 1.2 "AGC + loudness normalize").
/// It adapts only on speech-level blocks and over seconds, normalizing the microphone level
/// without flattening the loudness contrasts the emotion engine listens for.
class Agc {
public:
    struct Config {
        float target_dbfs = -23.0f;
        float max_gain_db = 18.0f;
        float min_gain_db = -12.0f;
        float attack_s = 3.0f;   ///< time constant when the gain must drop (input got louder)
        float release_s = 8.0f;  ///< time constant when the gain may rise
        float gate_dbfs = -50.0f;
        float limit = 0.95f;     ///< soft-limiter ceiling
    };

    Agc(Config config, int sample_rate);
    /// Processes one block in place.
    void process(std::span<float> block);
    [[nodiscard]] float gain_db() const noexcept { return gain_db_; }

private:
    Config cfg_;
    int sample_rate_;
    float gain_db_ = 0.0f;
};

/// Soft-knee limiter: transparent below 80 % of `ceiling`, tanh-compressed above it.
[[nodiscard]] float soft_limit(float x, float ceiling);

/// Neural noise suppression slot (RNNoise in the blueprint). In place, 16 kHz.
class INoiseSuppressor {
public:
    virtual ~INoiseSuppressor() = default;
    virtual void process(std::span<float> block) = 0;
};

/// Echo cancellation slot (WebRTC AEC3 in the blueprint): removes our own TTS, whose
/// playback reference arrives from 5.1, from the microphone signal. In place, 16 kHz.
class IEchoCanceller {
public:
    virtual ~IEchoCanceller() = default;
    virtual void process(std::span<float> mic, std::span<const float> reference) = 0;
};

/// "none" returns nullptr (pass-through). Throws ConfigError for engines not built in.
[[nodiscard]] std::unique_ptr<INoiseSuppressor> make_noise_suppressor(std::string_view name);
[[nodiscard]] std::unique_ptr<IEchoCanceller> make_echo_canceller(std::string_view name);

}  // namespace ee
