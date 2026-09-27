#pragma once

#include <cstdint>
#include <deque>
#include <map>
#include <string>

#include "core/runtime/model_registry.hpp"
#include "core/runtime/stage.hpp"
#include "core/tts/clause_chunker.hpp"
#include "core/tts/tts_engine.hpp"

namespace ee {

/// Stage 4.2 "Expressive TTS". Splits each Speech request at clause boundaries, synthesizes the
/// first clause immediately (first audio before the sentence ends) and the rest one per tick,
/// and streams the audio in ~100 ms SynthAudio chunks. Conditions on the style vector and the
/// speaker's voice print, drops queued clauses on barge-in, and publishes a neutral calibration
/// render at start-up so 5.2 can score relative to the voice's own baseline.
class TtsStage final : public IStage {
public:
    void open(StageContext& ctx) override;
    void process(Frame& frame) override;
    void tick() override;
    void close() override;

private:
    struct Job {
        Frame speech;  // header, prosody, style, source emotion
        Clause clause;
        std::uint32_t index = 0;
        bool last = false;
    };
    struct SpeakerVoice {
        SpeakerEmbedding print{};
        float f0 = 0.0f;  ///< median F0 in Hz, 0 if unknown
    };
    void synthesize(const Job& job);
    void emit_audio(const Job& job, std::uint32_t extra_flags);

    StageContext* ctx_ = nullptr;
    HotSwap<ITtsEngine> engine_;
    ChunkerConfig chunker_;
    std::size_t chunk_samples_ = 2400;
    std::deque<Job> pending_;
    std::map<std::uint64_t, SpeakerVoice> voices_;
    SpeakerVoice latest_voice_{};
    bool have_voice_ = false;
    bool use_voice_print_ = true;
    SynthesisResult result_;
};

}  // namespace ee
