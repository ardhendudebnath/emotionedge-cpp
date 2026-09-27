#pragma once

#include <memory>
#include <span>

#include "core/emotion/emotion_types.hpp"
#include "core/emotion/prosody_features.hpp"
#include "core/runtime/model_registry.hpp"
#include "core/runtime/params.hpp"

namespace ee {

/// One modality's opinion: a V·A·D point with a separate confidence per axis, because each
/// modality is trustworthy on different axes (prosody on arousal, words on valence).
struct ModalityEstimate {
    Vad vad;
    Vad confidence;  ///< per-axis, [0, 1]
    bool valid = false;
};

/// Acoustic emotion slot (blueprint 2.2 "ACOUSTIC: emotion2vec").
class IAcousticEmotionModel {
public:
    virtual ~IAcousticEmotionModel() = default;
    /// Estimates the emotion of a window or utterance of 16 kHz audio with its prosody frames.
    [[nodiscard]] virtual ModalityEstimate estimate(std::span<const float> audio,
                                                    std::span<const ProsodyFrame> frames, float hop_s) = 0;
    /// Called with each finished utterance's prosody so the model can adapt to the speaker.
    virtual void end_utterance(const ProsodySummary& summary) { (void)summary; }
    /// Sets the neutral reference directly (the TTS voice's calibration render, for 5.2).
    virtual void calibrate(const ProsodySummary& neutral) { (void)neutral; }
};

/// Transparent prosody rules (Scherer-style vocal cues), the v1 stand-in until the emotion2vec
/// head lands in phase 2:
///  - arousal: pitch range, speaking rate, spectral tilt (vocal effort), loudness and pitch level
///    relative to the speaker;
///  - dominance: falling final contours and loudness up, pauses and jitter down;
///  - valence: prosody alone barely separates joy from anger, so this model abstains
///    (near-zero confidence) and leaves valence to the lexical path.
class ProsodyEmotionModel final : public IAcousticEmotionModel {
public:
    [[nodiscard]] ModalityEstimate estimate(std::span<const float> audio, std::span<const ProsodyFrame> frames,
                                            float hop_s) override;
    void end_utterance(const ProsodySummary& summary) override;
    void calibrate(const ProsodySummary& neutral) override;

    /// The rule mapping itself, exposed for tests.
    [[nodiscard]] ModalityEstimate from_summary(const ProsodySummary& s) const;

private:
    // Speaker baseline (slow running means); unset until the first utterance.
    float base_f0_hz_ = 0.0f;
    float base_energy_db_ = 0.0f;
    bool has_baseline_ = false;
};

/// `acoustic: prosody` (default) or `onnx` (an emotion2vec + V·A·D head export, see ml/export).
[[nodiscard]] std::unique_ptr<IAcousticEmotionModel> make_acoustic_model(const Params& params,
                                                                         const ModelRegistry* registry);

}  // namespace ee
