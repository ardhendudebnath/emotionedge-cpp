#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <span>
#include <string>

#include "core/runtime/model_registry.hpp"
#include "core/runtime/params.hpp"

namespace ee {

/// Frame-level speech/silence decision (blueprint 1.3; Silero VAD in production).
class ISpeechDetector {
public:
    virtual ~ISpeechDetector() = default;
    /// Samples per decision (at the pipeline rate).
    [[nodiscard]] virtual std::size_t window() const noexcept = 0;
    /// Speech probability for exactly window() samples.
    virtual float probability(std::span<const float> window) = 0;
    virtual void reset() = 0;
};

/// Model-free detector: frame energy against an adaptive noise floor (the minimum over the last
/// few seconds). Good on clean input and for tests; Silero is the production choice.
class EnergySpeechDetector final : public ISpeechDetector {
public:
    struct Config {
        /// 32 ms at 16 kHz: "speech / silence every ~30 ms", the same windows as Silero, so the
        /// 160 ms hangover is exactly five windows with either detector.
        std::size_t window = 512;
        float snr_mid_db = 10.0f;        ///< SNR at which p = 0.5
        float snr_slope_db = 2.5f;
        float abs_floor_dbfs = -55.0f;   ///< quieter than this is never speech
        float noise_memory_s = 3.0f;
        int sample_rate = 16000;
    };

    explicit EnergySpeechDetector(Config config);
    [[nodiscard]] std::size_t window() const noexcept override { return cfg_.window; }
    float probability(std::span<const float> window) override;
    void reset() override;

private:
    Config cfg_;
    std::array<float, 256> history_{};
    std::size_t history_len_;
    std::size_t count_ = 0;
    std::size_t next_ = 0;
};

/// `detector: energy | silero` (+ `model`/`model_id`, `window_ms` for energy).
[[nodiscard]] std::unique_ptr<ISpeechDetector> make_speech_detector(const Params& params,
                                                                    const ModelRegistry* registry,
                                                                    int sample_rate);

#if defined(EE_HAVE_ONNXRUNTIME)
/// Silero VAD v5/v6 ONNX (512-sample windows at 16 kHz with 64 samples of context).
[[nodiscard]] std::unique_ptr<ISpeechDetector> make_silero_detector(const std::string& model_path,
                                                                    const Params& params,
                                                                    const ModelRegistry* registry,
                                                                    int sample_rate);
#endif

}  // namespace ee
