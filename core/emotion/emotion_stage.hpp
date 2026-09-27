#pragma once

#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <vector>

#include "core/emotion/acoustic.hpp"
#include "core/emotion/fusion.hpp"
#include "core/emotion/lexical.hpp"
#include "core/emotion/prosody_features.hpp"
#include "core/runtime/stage.hpp"

namespace ee {

/// Stage 2.2 "Emotion Engine". While the speaker talks it emits EMA-smoothed partial estimates
/// every `hop_ms` from the last `window_ms` of prosody fused with the latest ASR text. At the
/// endpoint it fuses the whole utterance acoustically (in parallel with ASR's final decode), and
/// once the final transcript arrives it emits the final Emotion frame with the utterance's
/// energy envelope, which the state tracker aligns with word timestamps to find emphasis.
class EmotionEngineStage final : public IStage {
public:
    void open(StageContext& ctx) override;
    void process(Frame& frame) override;
    void tick() override;
    void close() override;

private:
    struct Utterance {
        std::uint64_t id = 0;
        ProsodyTracker prosody;
        std::vector<float> audio;
        std::size_t next_estimate_frame = 0;
        bool endpoint = false;
        bool final_text = false;
        TimePoint endpoint_seen{};
        ModalityEstimate lexical;
        ModalityEstimate acoustic_final;
        Frame header;
    };
    Utterance* record(std::uint64_t id);
    void on_audio(Frame& f);
    void on_transcript(const Frame& f);
    void try_finalize(Utterance& u, bool force);
    void emit_state(const Utterance& u, const EmotionState& state, const Vad& axes, bool final);

    StageContext* ctx_ = nullptr;
    std::unique_ptr<IAcousticEmotionModel> acoustic_;
    std::unique_ptr<ILexicalEmotionModel> lexical_;
    FusionConfig fusion_;
    EmotionSmoother smoother_;
    int rate_ = 16000;
    std::size_t window_frames_ = 100;
    std::size_t hop_frames_ = 50;
    std::size_t max_audio_ = 0;
    float final_alpha_ = 0.8f;
    bool wait_for_transcript_ = true;
    double transcript_timeout_s_ = 1.5;
    std::deque<Utterance> utterances_;
    std::deque<std::uint64_t> finished_;
};

}  // namespace ee
