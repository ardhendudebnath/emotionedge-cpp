#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "core/audio/audio_io.hpp"
#include "core/audio/resampler.hpp"
#include "core/runtime/stage.hpp"

namespace ee {

/// Joins synthesized chunks into one continuous stream. Chunks inside a clause are contiguous
/// samples and are concatenated as-is; at a clause boundary the last `crossfade` samples are
/// held back and overlapped with the next clause's first samples (equal-power fade), which
/// hides the click between separately synthesized clauses.
class ChunkJoiner {
public:
    explicit ChunkJoiner(std::size_t crossfade_samples) : crossfade_(crossfade_samples) {}
    /// Appends playout-ready samples to `out`.
    void push(std::span<const float> chunk, bool clause_end, std::vector<float>& out);
    /// Releases any held tail (end of utterance).
    void finish(std::vector<float>& out);
    void reset() { held_.clear(); }

private:
    std::size_t crossfade_;
    std::vector<float> held_;
};

/// Stage 5.1 "Playback". Live mode keeps a jitter buffer (60 ms by default) before an
/// utterance starts, writes to AudioIo::playback, feeds the played signal back to the AEC as
/// its reference and raises `playback_active` for barge-in. Offline mode places each utterance
/// on AudioIo::timeline where it would have played live: after the source speech ended plus the
/// nominal pipeline latency, never overlapping the previous utterance. Emits a Playout frame
/// per utterance for captions.
class PlaybackStage final : public IStage {
public:
    void open(StageContext& ctx) override;
    void process(Frame& frame) override;
    void tick() override;
    void close() override;

private:
    enum class Mode { Offline, Live, Discard };
    void begin_utterance(const Frame& f);
    void deliver(bool final_chunk);
    void finish_utterance();
    /// Tells the TTS how far playout runs behind (AudioIo::playout_delay).
    void publish_backlog();
    void feed_echo_reference(std::span<const float> played);

    StageContext* ctx_ = nullptr;
    AudioIo* io_ = nullptr;
    Mode mode_ = Mode::Discard;
    int out_rate_ = 24000;
    std::size_t jitter_samples_ = 0;
    double offline_latency_s_ = 0.735;
    double gap_s_ = 0.15;
    std::unique_ptr<ChunkJoiner> joiner_;
    std::unique_ptr<Resampler> resampler_;
    std::unique_ptr<Resampler> echo_resampler_;
    std::vector<float> converted_;
    std::vector<float> ready_;
    std::vector<float> echo_;

    std::uint64_t utterance_ = 0;
    std::uint64_t cancel_before_ = 0;
    bool started_ = false;
    bool first_audio_marked_ = false;
    bool utterance_final_ = false;
    int in_rate_ = 0;
    double src_start_ = 0.0;
    double src_end_ = 0.0;
    double out_start_ = 0.0;
    double write_pos_ = 0.0;
    std::int64_t written_ = 0;
};

}  // namespace ee
