#include "core/audio/frontend.hpp"

#include <algorithm>
#include <cmath>

#include "core/audio/wav.hpp"
#include "core/runtime/log.hpp"

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
    echo_ = make_echo_canceller(p.str("echo_canceller", "none"), p.sub("aec"), rate_);
    // It needs the played reference, which only real-time and live runs have.
    if (io_->echo_reference == nullptr) echo_.reset();
    record_ = p.str("record");
    // The reference is aligned for a canceller, or recorded with the microphone (data for one).
    use_reference_ = io_->echo_reference != nullptr && (echo_ || !record_.empty());
    if (use_reference_ && io_->playback != nullptr && io_->playback->sample_rate() != rate_) {
        ref_resampler_ = std::make_unique<Resampler>(io_->playback->sample_rate(), rate_);
    }
    ref_slack_ms_ = p.integer("aec.reference_slack_ms", 40);

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

void FrontendStage::align_reference(std::size_t n) {
    SpscRing<float>& tap = *io_->echo_reference;
    const int ref_rate = ref_resampler_ ? ref_resampler_->in_rate() : rate_;
    const bool first = !ref_started_;
    if (first) {
        // Start in step with the microphone, keeping only the last reference_slack of what
        // already played. That slack absorbs the jitter between the playback callback and this
        // thread, so the reference is never late, which would mean a jump in the echo delay.
        // It must stay shorter than the echo's own delay (device buffers plus flight), or the
        // reference would come after the echo it explains. Expect >= 40 ms on real hardware.
        const auto slack = static_cast<std::size_t>(ref_slack_ms_ * ref_rate / 1000);
        const std::size_t available = tap.read_available();
        tap.skip(available > slack ? available - slack : 0);
        ref_started_ = true;
    }
    // Lockstep: n reference samples for n microphone samples. Both streams are continuous (the
    // tap records silence too), so the offset between them stays fixed.
    while (ref_fifo_.size() < n + ref_debt_) {
        const std::size_t available = tap.read_available();
        if (available == 0) break;
        const std::size_t missing = n + ref_debt_ - ref_fifo_.size();
        const auto want = static_cast<std::size_t>(static_cast<std::int64_t>(missing) * ref_rate / rate_ + 1);
        ref_in_.resize(std::min(available, want));
        (void)tap.read(ref_in_);
        if (ref_resampler_) {
            ref_resampler_->process(ref_in_, ref_fifo_);
        } else {
            ref_fifo_.insert(ref_fifo_.end(), ref_in_.begin(), ref_in_.end());
        }
    }
    // Reference that was late earlier was replaced by silence: skip it now that it is here.
    const std::size_t repay = std::min(ref_debt_, ref_fifo_.size() > n ? ref_fifo_.size() - n : 0);
    ref_fifo_.erase(ref_fifo_.begin(), ref_fifo_.begin() + static_cast<std::ptrdiff_t>(repay));
    ref_debt_ -= repay;
    const std::size_t got = std::min(n, ref_fifo_.size());
    reference_.assign(ref_fifo_.begin(), ref_fifo_.begin() + static_cast<std::ptrdiff_t>(got));
    reference_.resize(n, 0.0f);
    ref_fifo_.erase(ref_fifo_.begin(), ref_fifo_.begin() + static_cast<std::ptrdiff_t>(got));
    // Just after the skip, a short reference is not late: nothing was owed yet.
    if (!first && got < n) {
        ref_debt_ += n - got;
        ++ref_late_;
    }
}

void FrontendStage::process_block(std::span<float> block, std::size_t samples_after) {
    if (!record_.empty()) rec_mic_.insert(rec_mic_.end(), block.begin(), block.end());
    if (use_reference_) {
        align_reference(block.size());
        if (!record_.empty()) rec_ref_.insert(rec_ref_.end(), reference_.begin(), reference_.end());
    }
    if (echo_) {
        const bool playing = rms_dbfs(reference_) > -50.0f;
        double before = 0.0;
        if (playing) {
            for (float x : block) before += static_cast<double>(x) * x;
        }
        echo_->process(block, reference_);
        if (playing) {
            echo_in_energy_ += before;
            for (float x : block) echo_out_energy_ += static_cast<double>(x) * x;
            echo_samples_ += block.size();
        }
        if (!record_.empty()) rec_aec_.insert(rec_aec_.end(), block.begin(), block.end());
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

void FrontendStage::close() {
    if (echo_samples_ > 0 && echo_in_energy_ > 0.0) {
        // Near-end speech over the playback counts too, so this understates the echo removed.
        const double in_db = 10.0 * std::log10(echo_in_energy_ / static_cast<double>(echo_samples_) + 1e-12);
        const double out_db = 10.0 * std::log10(echo_out_energy_ / static_cast<double>(echo_samples_) + 1e-12);
        log::info("frontend: while the translation played (", static_cast<double>(echo_samples_) / rate_,
                  " s), the echo canceller took the microphone from ", in_db, " to ", out_db, " dBFS (",
                  in_db - out_db, " dB)");
    }
    if (ref_late_ > 0) {
        log::warn("frontend: the echo reference was late ", ref_late_,
                  " times (each one a jump in the echo delay); raise aec.reference_slack_ms");
    }
    if (!record_.empty()) {
        write_wav(record_ + "_mic.wav", rec_mic_, rate_);
        if (use_reference_) write_wav(record_ + "_ref.wav", rec_ref_, rate_);
        if (echo_) write_wav(record_ + "_aec.wav", rec_aec_, rate_);
    }
}

}  // namespace ee
