#include "core/audio/speech_detector.hpp"

#include <algorithm>
#include <cmath>

#include "core/audio/dsp.hpp"

namespace ee {

EnergySpeechDetector::EnergySpeechDetector(Config config) : cfg_(config) {
    const double windows_per_s = static_cast<double>(cfg_.sample_rate) / static_cast<double>(cfg_.window);
    history_len_ = std::clamp<std::size_t>(static_cast<std::size_t>(cfg_.noise_memory_s * windows_per_s), 4,
                                           history_.size());
}

void EnergySpeechDetector::reset() {
    count_ = 0;
    next_ = 0;
}

float EnergySpeechDetector::probability(std::span<const float> window) {
    const float level = rms_dbfs(window);
    history_[next_] = level;
    next_ = (next_ + 1) % history_len_;
    count_ = std::min(count_ + 1, history_len_);

    // Noise floor = quietest recent window; speech has syllable dips, so this tracks the noise.
    float noise = level;
    for (std::size_t i = 0; i < count_; ++i) noise = std::min(noise, history_[i]);
    const float snr = level - noise;
    const auto sigmoid = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };
    return sigmoid((snr - cfg_.snr_mid_db) / cfg_.snr_slope_db) * sigmoid((level - cfg_.abs_floor_dbfs) / 2.0f);
}

std::unique_ptr<ISpeechDetector> make_speech_detector(const Params& params, const ModelRegistry* registry,
                                                      int sample_rate) {
    const std::string kind = params.str("detector", "energy");
    if (kind == "energy") {
        EnergySpeechDetector::Config cfg;
        cfg.sample_rate = sample_rate;
        cfg.window = static_cast<std::size_t>(sample_rate * params.integer("window_ms", 32) / 1000);
        cfg.snr_mid_db = params.real("snr_db", cfg.snr_mid_db);
        cfg.abs_floor_dbfs = params.real("floor_dbfs", cfg.abs_floor_dbfs);
        return std::make_unique<EnergySpeechDetector>(cfg);
    }
    if (kind == "silero") {
#if defined(EE_HAVE_ONNXRUNTIME)
        const std::string model = resolve_model_path(params, registry, "model");
        if (model.empty()) throw ConfigError("detector 'silero' needs a 'model' path or 'model_id'");
        return make_silero_detector(model, params, registry, sample_rate);
#else
        (void)registry;
        throw ConfigError("detector 'silero' needs a build with -DEE_WITH_ONNXRUNTIME=ON");
#endif
    }
    throw ConfigError("unknown speech detector '" + kind + "' (energy | silero)");
}

}  // namespace ee
