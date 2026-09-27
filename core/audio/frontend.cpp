#include "core/audio/frontend.hpp"

#include <algorithm>

namespace ee {

void FrontendStage::open(StageContext& ctx) {
    ctx_ = &ctx;
    io_ = ctx.services().audio;
    if (io_ == nullptr || io_->capture == nullptr) {
        throw ConfigError("frontend needs an audio capture source (AudioIo::capture)");
    }
    source_ = io_->capture;
    const Params& p = ctx.params();
    rate_ = ctx.pipeline().sample_rate;
    frame_ = static_cast<std::size_t>(rate_ * p.integer("frame_ms", 20) / 1000);
    if (frame_ == 0) throw ConfigError("frontend frame_ms is too small");
    if (source_->sample_rate() != rate_) resampler_ = std::make_unique<Resampler>(source_->sample_rate(), rate_);

    if (p.flag("agc.enabled", true)) {
        Agc::Config cfg;
        cfg.target_dbfs = p.real("agc.target_dbfs", cfg.target_dbfs);
        cfg.max_gain_db = p.real("agc.max_gain_db", cfg.max_gain_db);
        cfg.min_gain_db = p.real("agc.min_gain_db", cfg.min_gain_db);
        cfg.attack_s = p.real("agc.attack_s", cfg.attack_s);
        cfg.release_s = p.real("agc.release_s", cfg.release_s);
        agc_ = std::make_unique<Agc>(cfg, rate_);
    }
    noise_ = make_noise_suppressor(p.str("noise_suppressor", "none"));
    echo_ = make_echo_canceller(p.str("echo_canceller", "none"));

    const auto input_frame = static_cast<std::size_t>(
        (static_cast<std::int64_t>(frame_) * source_->sample_rate() + rate_ - 1) / rate_);
    read_buf_.resize(input_frame);
    ready_.reserve(frame_ * 8);
    reference_.resize(frame_);
}

void FrontendStage::tick() {
    if (finished_) return;
    // Deterministic runs pull one frame per pass (a file then streams like a live input);
    // live runs drain whatever the capture callback has delivered.
    const bool deterministic = ctx_->deterministic();
    for (int reads = 0; reads < (deterministic ? 1 : 64); ++reads) {
        const std::size_t n = source_->read(read_buf_);
        if (n == 0) break;
        const std::span<const float> in(read_buf_.data(), n);
        if (resampler_) {
            resampler_->process(in, ready_);
        } else {
            ready_.insert(ready_.end(), in.begin(), in.end());
        }
    }
    const bool done = source_->exhausted();
    if (done && resampler_) resampler_->flush(ready_);
    emit_ready(done);
    if (done) {
        finished_ = true;
        ctx_->finish();
    }
}

void FrontendStage::emit_ready(bool flush_tail) {
    std::size_t offset = 0;
    while (ready_.size() - offset >= frame_ || (flush_tail && offset < ready_.size())) {
        const std::size_t n = std::min(frame_, ready_.size() - offset);
        const std::size_t after = ready_.size() - offset - n;
        process_block(std::span<float>(ready_.data() + offset, n), after);
        offset += n;
    }
    ready_.erase(ready_.begin(), ready_.begin() + static_cast<std::ptrdiff_t>(offset));
}

void FrontendStage::process_block(std::span<float> block, std::size_t samples_after) {
    if (io_->echo_reference != nullptr) {
        reference_.resize(block.size());
        const std::size_t got = io_->echo_reference->read(reference_);
        std::fill(reference_.begin() + static_cast<std::ptrdiff_t>(got), reference_.end(), 0.0f);
        if (echo_) echo_->process(block, reference_);
    }
    if (noise_) noise_->process(block);
    if (agc_) agc_->process(block);

    Frame& f = ctx_->make(FrameKind::Audio);
    f.stream_pos = produced_;
    f.sample_rate = rate_;
    TimePoint origin = ctx_->now();
    if (!ctx_->deterministic()) {
        // Back-date to the capture time of this block's first sample.
        const double queued = static_cast<double>(source_->backlog()) / source_->sample_rate() +
                              static_cast<double>(samples_after + block.size()) / rate_;
        origin = add_seconds(origin, -queued);
    }
    f.t_origin = origin;
    f.audio.assign(block.begin(), block.end());
    produced_ += static_cast<std::int64_t>(block.size());
    ctx_->emit(f);
}

}  // namespace ee
