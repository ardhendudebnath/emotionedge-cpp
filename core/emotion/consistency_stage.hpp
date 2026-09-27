#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/audio/resampler.hpp"
#include "core/emotion/acoustic.hpp"
#include "core/emotion/fusion.hpp"
#include "core/emotion/lexical.hpp"
#include "core/runtime/stage.hpp"
#include "core/telemetry/metrics.hpp"

namespace ee {

/// Stage 5.2 "Emotion Consistency" (closed loop). Re-scores every synthesized clause with the
/// same emotion engine used on the input, calibrated on the TTS voice's neutral render, computes
/// ECS = 1 - ‖VAD_src - VAD_out‖ / 2√3, logs it to telemetry and sends a Feedback frame
/// (score, ΔVAD, per-axis confidence of the measurement) back to the emotion controller (4.1).
class EmotionConsistencyStage final : public IStage {
public:
    void open(StageContext& ctx) override;
    void process(Frame& frame) override;

    [[nodiscard]] float last_score() const noexcept { return last_score_; }

private:
    void evaluate_clause(const Frame& last_chunk);

    StageContext* ctx_ = nullptr;
    std::unique_ptr<IAcousticEmotionModel> acoustic_;
    std::unique_ptr<ILexicalEmotionModel> lexical_;
    FusionConfig fusion_;
    float threshold_ = 0.75f;
    int rate_ = 16000;
    std::unique_ptr<Resampler> resampler_;
    int resampler_in_rate_ = 0;
    std::vector<float> clause_;
    std::string clause_text_;
    std::uint64_t utterance_ = 0;
    std::uint32_t clause_index_ = 0;
    float last_score_ = 1.0f;
    telemetry::Counter* below_threshold_ = nullptr;
};

}  // namespace ee
