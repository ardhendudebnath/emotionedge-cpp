#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/audio/audio_io.hpp"
#include "core/audio/dsp.hpp"
#include "core/audio/resampler.hpp"
#include "core/runtime/stage.hpp"

namespace ee {

/// Stages 1.1 + 1.2: pulls captured audio (AudioIo::capture), converts it to the pipeline rate,
/// runs echo cancellation, noise suppression and AGC, and emits 20 ms Audio frames stamped
/// with their capture time. It is the graph's source: in deterministic runs it reads one frame
/// per scheduler pass, which replays a file as if it were streaming.
class FrontendStage final : public IStage {
public:
    void open(StageContext& ctx) override;
    void process(Frame& frame) override { (void)frame; }
    void tick() override;
    void close() override;

private:
    void emit_ready(bool flush_tail);
    void process_block(std::span<float> block, std::size_t samples_after);
    /// Fills reference_ with the played audio that is time-aligned with the next n microphone
    /// samples (AudioIo::echo_reference, resampled to the pipeline rate).
    void align_reference(std::size_t n);

    StageContext* ctx_ = nullptr;
    AudioIo* io_ = nullptr;
    IAudioSource* source_ = nullptr;
    std::unique_ptr<Resampler> resampler_;
    std::unique_ptr<Agc> agc_;
    std::unique_ptr<INoiseSuppressor> noise_;
    std::unique_ptr<IEchoCanceller> echo_;
    int rate_ = 16000;
    std::size_t frame_ = 320;
    std::vector<float> read_buf_;
    std::vector<float> ready_;
    std::vector<float> reference_;
    std::int64_t produced_ = 0;
    bool finished_ = false;

    // Echo reference, consumed in lockstep with the microphone (for the canceller, or recorded).
    bool use_reference_ = false;
    std::unique_ptr<Resampler> ref_resampler_;
    std::vector<float> ref_in_;
    std::vector<float> ref_fifo_;  ///< at the pipeline rate
    std::size_t ref_debt_ = 0;     ///< reference that arrived late: skipped when it comes
    std::size_t ref_late_ = 0;     ///< how often that happened
    std::int64_t ref_slack_ms_ = 40;
    bool ref_started_ = false;
    // Microphone energy before / after the canceller while the translation plays.
    double echo_in_energy_ = 0.0;
    double echo_out_energy_ = 0.0;
    std::size_t echo_samples_ = 0;
    // `record: <prefix>` writes <prefix>_mic.wav at close, plus the aligned reference
    // (_ref.wav) when audio plays, and the canceller's output (_aec.wav) when there is one.
    std::string record_;
    std::vector<float> rec_mic_;
    std::vector<float> rec_ref_;
    std::vector<float> rec_aec_;
};

}  // namespace ee
