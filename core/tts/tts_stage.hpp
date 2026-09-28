#pragma once

#include <cstdint>
#include <deque>
#include <map>
#include <string>

#include "core/runtime/model_registry.hpp"
#include "core/runtime/stage.hpp"
#include "core/telemetry/metrics.hpp"
#include "core/tts/clause_chunker.hpp"
#include "core/tts/tts_engine.hpp"

namespace ee {

/// Adaptive pacing (phase 4). A translation that is longer than its source, as Hindi is than
/// English, queues up behind the previous one when the speaker talks on. Each utterance's
/// speaking rate is raised, up to `max_speed`, by whichever asks more:
/// - backlog: the time it would wait before playing, from 1x at `start_s` to the cap at `full_s`;
/// - fit (`fit_next`): it should end before the next utterance's translation is due, about
///   the speaker's usual gap plus an utterance like this one later. So speed >= predicted
///   duration / (source duration + expected gap). A speaker who pauses leaves room and is not
///   sped up.
/// The rate is fixed per utterance, so it does not change mid-sentence.
struct PacingConfig {
    float start_s = 0.25f;
    float full_s = 1.5f;
    float max_speed = 1.0f;  ///< 1 = off
    bool fit_next = false;

    /// Speed multiplier for an utterance that would wait `backlog_s` behind earlier audio and
    /// last `predicted_s` (at its planned rate), for `source_s` of source speech followed by an
    /// expected `gap_s` of silence.
    [[nodiscard]] float speed(double backlog_s, double predicted_s = 0.0, double source_s = 0.0,
                              double gap_s = 0.0) const noexcept;
};

/// Stage 4.2 "Expressive TTS". Splits each Speech request at clause boundaries, synthesizes the
/// first clause immediately (first audio before the sentence ends) and the rest one per tick,
/// and streams the audio in ~100 ms SynthAudio chunks. Conditions on the style vector and the
/// speaker's voice print, drops queued clauses on barge-in, paces each utterance by the
/// playout backlog (PacingConfig), and publishes a neutral calibration render at start-up so
/// 5.2 can score relative to the voice's own baseline.
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
        float pace = 1.0f;  ///< adaptive pacing speed-up, the same for every clause of an utterance
    };
    struct SpeakerVoice {
        SpeakerEmbedding print{};
        float f0 = 0.0f;  ///< median F0 in Hz, 0 if unknown
    };
    void synthesize(const Job& job);
    void emit_audio(const Job& job, std::uint32_t extra_flags);
    /// Seconds a new utterance would wait before playing: playout's backlog plus the clauses
    /// still waiting here.
    [[nodiscard]] double backlog_s(const Frame& speech) const;

    StageContext* ctx_ = nullptr;
    HotSwap<ITtsEngine> engine_;
    ChunkerConfig chunker_;
    PacingConfig pacing_;
    /// Recent audio seconds per character of text at speed 1 (the calibration render seeds it).
    double seconds_per_char_ = 0.0;
    /// The speaker's recent gap between utterances (pacing's fit term), and the last source end.
    double gap_s_ = 1.0;
    double last_src_end_ = -1.0;
    telemetry::Histogram* pace_metric_ = nullptr;
    std::size_t chunk_samples_ = 2400;
    std::deque<Job> pending_;
    std::map<std::uint64_t, SpeakerVoice> voices_;
    SpeakerVoice latest_voice_{};
    bool have_voice_ = false;
    bool use_voice_print_ = true;
    SynthesisResult result_;
};

}  // namespace ee
