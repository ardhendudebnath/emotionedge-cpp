#pragma once

#include <cstdint>
#include <memory>
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

private:
    void emit_ready(bool flush_tail);
    void process_block(std::span<float> block, std::size_t samples_after);

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
};

}  // namespace ee
