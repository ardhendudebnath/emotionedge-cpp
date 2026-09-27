#include "core/emotion/acoustic.hpp"

#include <algorithm>
#include <cmath>
#include <string>

namespace ee {

#if defined(EE_HAVE_ONNXRUNTIME)
std::unique_ptr<IAcousticEmotionModel> make_onnx_acoustic_model(const std::string& path, const Params& params,
                                                                const ModelRegistry* registry);
#endif

namespace {
float clamp_z(float z) { return std::clamp(z, -3.0f, 3.0f); }

// Population priors for read speech, used until the speaker baseline exists. They are the
// v1 calibration and should be refit on IEMOCAP / MSP-Podcast in the ml/ factory (P2).
constexpr float kPriorRangeSt = 4.0f;
constexpr float kPriorRangeScale = 2.5f;
constexpr float kPriorSyllableRate = 4.0f;
constexpr float kPriorRateScale = 1.5f;
constexpr float kPriorHfDb = -13.0f;
constexpr float kPriorHfScale = 3.0f;
}  // namespace

ModalityEstimate ProsodyEmotionModel::estimate(std::span<const float> audio, std::span<const ProsodyFrame> frames,
                                               float hop_s) {
    (void)audio;
    return from_summary(summarize(frames, hop_s));
}

ModalityEstimate ProsodyEmotionModel::from_summary(const ProsodySummary& s) const {
    ModalityEstimate e;
    if (s.voiced_frames < 8) return e;  // too little voicing to judge

    // Level-independent cues against population priors.
    const float range_z = clamp_z((s.f0_range_st - kPriorRangeSt) / kPriorRangeScale);
    const float rate_z = clamp_z((s.syllable_rate - kPriorSyllableRate) / kPriorRateScale);
    const float tilt_z = clamp_z((s.hf_db - kPriorHfDb) / kPriorHfScale);
    // Speaker-relative cues: zero until a baseline exists.
    const float energy_rel = has_baseline_ ? clamp_z((s.energy_db - base_energy_db_) / 4.0f) : 0.0f;
    const float pitch_rel =
        has_baseline_ && base_f0_hz_ > 0.0f ? clamp_z(semitones(s.f0_median_hz, base_f0_hz_) / 2.0f) : 0.0f;
    // Contour and fluency cues for dominance.
    const float fall_z = clamp_z(-s.final_slope_st_s / 6.0f);
    const float pause_z = clamp_z((s.pause_ratio - 0.2f) / 0.15f);
    const float jitter_z = clamp_z((s.jitter - 0.02f) / 0.02f);

    const float arousal =
        std::tanh(0.45f * range_z + 0.35f * rate_z + 0.45f * tilt_z + 0.5f * energy_rel + 0.35f * pitch_rel);
    const float dominance =
        std::tanh(0.5f * fall_z + 0.3f * energy_rel + 0.25f * tilt_z - 0.35f * pause_z - 0.25f * jitter_z);

    // Confidence grows with the amount of voicing and with how far the cues sit from neutral.
    const float coverage = std::min(1.0f, static_cast<float>(s.voiced_frames) / 40.0f);
    e.vad = {0.0f, arousal, dominance};
    e.confidence = {0.05f, coverage * 0.8f * (0.6f + 0.4f * std::abs(arousal)),
                    coverage * 0.5f * (0.6f + 0.4f * std::abs(dominance))};
    e.valid = true;
    return e;
}

void ProsodyEmotionModel::end_utterance(const ProsodySummary& s) {
    if (s.voiced_frames < 8) return;
    if (!has_baseline_) {
        calibrate(s);
        return;
    }
    constexpr float kRate = 0.2f;  // a few utterances to adapt
    base_f0_hz_ += kRate * (s.f0_median_hz - base_f0_hz_);
    base_energy_db_ += kRate * (s.energy_db - base_energy_db_);
}

void ProsodyEmotionModel::calibrate(const ProsodySummary& neutral) {
    if (neutral.voiced_frames < 8) return;
    base_f0_hz_ = neutral.f0_median_hz;
    base_energy_db_ = neutral.energy_db;
    has_baseline_ = true;
}

std::unique_ptr<IAcousticEmotionModel> make_acoustic_model(const Params& params, const ModelRegistry* registry) {
    const std::string kind = params.str("acoustic", "prosody");
    if (kind == "prosody") return std::make_unique<ProsodyEmotionModel>();
    if (kind == "onnx") {
#if defined(EE_HAVE_ONNXRUNTIME)
        const std::string path = resolve_model_path(params, registry, "acoustic_model");
        if (path.empty()) throw ConfigError("acoustic 'onnx' needs 'acoustic_model' or 'acoustic_model_id'");
        return make_onnx_acoustic_model(path, params, registry);
#else
        (void)registry;
        throw ConfigError("acoustic 'onnx' needs a build with -DEE_WITH_ONNXRUNTIME=ON");
#endif
    }
    throw ConfigError("unknown acoustic emotion model '" + kind + "' (prosody | onnx)");
}

}  // namespace ee
