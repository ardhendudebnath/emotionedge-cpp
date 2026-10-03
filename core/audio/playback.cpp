#include "core/audio/playback.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

#include "core/runtime/log.hpp"
#include "core/telemetry/telemetry.hpp"

namespace ee {

// ---- ChunkJoiner ------------------------------------------------------------------------------

void ChunkJoiner::push(std::span<const float> chunk, bool clause_end, std::vector<float>& out) {
    std::size_t start = 0;
    if (!held_.empty()) {
        const std::size_t n = std::min(held_.size(), chunk.size());
        for (std::size_t i = 0; i < n; ++i) {
            // Equal-power crossfade: fade-out and fade-in gains keep constant total power.
            const float t = (static_cast<float>(i) + 0.5f) / static_cast<float>(n);
            const float fade_in = std::sin(t * std::numbers::pi_v<float> / 2.0f);
            const float fade_out = std::cos(t * std::numbers::pi_v<float> / 2.0f);
            out.push_back(held_[i] * fade_out + chunk[i] * fade_in);
        }
        out.insert(out.end(), held_.begin() + static_cast<std::ptrdiff_t>(n), held_.end());
        held_.clear();
        start = n;
    }
    const std::size_t remaining = chunk.size() - start;
    if (clause_end && crossfade_ > 0 && remaining > crossfade_) {
        const auto split = chunk.begin() + static_cast<std::ptrdiff_t>(chunk.size() - crossfade_);
        out.insert(out.end(), chunk.begin() + static_cast<std::ptrdiff_t>(start), split);
        held_.assign(split, chunk.end());
    } else {
        out.insert(out.end(), chunk.begin() + static_cast<std::ptrdiff_t>(start), chunk.end());
    }
}

void ChunkJoiner::finish(std::vector<float>& out) {
    out.insert(out.end(), held_.begin(), held_.end());
    held_.clear();
}

// ---- PlaybackStage ------------------------------------------------------------------------------

void PlaybackStage::open(StageContext& ctx) {
    ctx_ = &ctx;
    io_ = ctx.services().audio;
    const Params& p = ctx.params();
    const std::string mode = p.str("mode", "auto");
    if (mode == "offline" || (mode == "auto" && io_ != nullptr && io_->timeline != nullptr)) {
        if (io_ == nullptr || io_->timeline == nullptr) throw ConfigError("playback mode 'offline' needs AudioIo::timeline");
        mode_ = Mode::Offline;
        out_rate_ = io_->timeline->sample_rate();
    } else if (mode == "live" || (mode == "auto" && io_ != nullptr && io_->playback != nullptr)) {
        if (io_ == nullptr || io_->playback == nullptr) throw ConfigError("playback mode 'live' needs AudioIo::playback");
        mode_ = Mode::Live;
        out_rate_ = io_->playback->sample_rate();
    } else if (mode == "discard" || mode == "auto") {
        mode_ = Mode::Discard;
        out_rate_ = static_cast<int>(p.integer("sample_rate", 24000));
    } else {
        throw ConfigError("playback mode must be auto, offline, live or discard");
    }
    jitter_samples_ = static_cast<std::size_t>(out_rate_ * p.integer("jitter_ms", 60) / 1000);
    offline_latency_s_ = p.number("offline_latency_ms", 735.0) / 1000.0;
    gap_s_ = p.number("gap_ms", 150.0) / 1000.0;
    joiner_ = std::make_unique<ChunkJoiner>(static_cast<std::size_t>(out_rate_ * p.integer("crossfade_ms", 5) / 1000));
    if (io_ != nullptr) {
        io_->playout_queued_s.store(0.0);
        io_->playout_free_at_s.store(-1e9);
    }
}

void PlaybackStage::publish_backlog() {
    if (mode_ == Mode::Live) {
        const std::size_t queued = io_->playback->queued() + ready_.size();
        io_->playout_queued_s.store(static_cast<double>(queued) / out_rate_, std::memory_order_relaxed);
    } else if (mode_ == Mode::Offline) {
        // The next utterance starts at max(end + gap, its src_end + latency): see deliver().
        io_->playout_free_at_s.store(io_->timeline->end_seconds() + gap_s_ - offline_latency_s_,
                                     std::memory_order_relaxed);
    }
}

void PlaybackStage::begin_utterance(const Frame& f) {
    utterance_ = f.utterance;
    started_ = false;
    first_audio_marked_ = false;
    utterance_final_ = false;
    src_start_ = f.src_start;
    src_end_ = f.src_end;
    written_ = 0;
    ready_.clear();
    joiner_->reset();
    if (!resampler_ || in_rate_ != f.sample_rate) {
        in_rate_ = f.sample_rate;
        resampler_ = std::make_unique<Resampler>(in_rate_, out_rate_);
    } else {
        resampler_->reset();
    }
}

void PlaybackStage::process(Frame& f) {
    if (f.kind == FrameKind::Control && f.has(frame_flags::kBargeIn)) {
        // The user is talking: stop the current translation immediately.
        cancel_before_ = std::max(cancel_before_, f.utterance);
        ready_.clear();
        joiner_->reset();
        if (mode_ == Mode::Live) {
            io_->playback->flush();
            io_->playback->set_streaming(false);
            io_->playout_queued_s.store(0.0);
        }
        if (io_ != nullptr) io_->playback_active.store(false);
        return;
    }
    if (f.kind != FrameKind::SynthAudio || f.has(frame_flags::kCalibration)) return;
    if (f.utterance < cancel_before_ || f.sample_rate <= 0) return;
    if (f.utterance != utterance_ || in_rate_ == 0) begin_utterance(f);

    converted_.clear();
    resampler_->process(f.audio, converted_);
    if (f.is_final()) resampler_->flush(converted_);
    joiner_->push(converted_, f.has(frame_flags::kClauseEnd) && !f.is_final(), ready_);
    if (f.is_final()) {
        joiner_->finish(ready_);
        utterance_final_ = true;
    }
    deliver(f.is_final());
}

void PlaybackStage::deliver(bool final_chunk) {
    auto* telemetry = ctx_->services().telemetry;
    switch (mode_) {
    case Mode::Offline: {
        if (!started_) {
            out_start_ = std::max(io_->timeline->end_seconds() + (io_->timeline->end_seconds() > 0 ? gap_s_ : 0.0),
                                  src_end_ + offline_latency_s_);
            write_pos_ = out_start_;
            started_ = true;
        }
        if (!ready_.empty()) {
            io_->timeline->write_at(write_pos_, ready_);
            write_pos_ += static_cast<double>(ready_.size()) / out_rate_;
            written_ += static_cast<std::int64_t>(ready_.size());
            ready_.clear();
            if (!first_audio_marked_) {
                telemetry->mark(utterance_, telemetry::Milestone::FirstAudio, ctx_->now());
                first_audio_marked_ = true;
            }
        }
        break;
    }
    case Mode::Live: {
        if (!started_ && ready_.size() < jitter_samples_ && !final_chunk) return;  // jitter buffer
        if (!started_) {
            started_ = true;
            out_start_ = static_cast<double>(written_) / out_rate_;
        }
        if (!ready_.empty()) {
            const std::size_t accepted = io_->playback->write(ready_);
            if (accepted > 0) {
                io_->playback_active.store(true);
                io_->playback->set_streaming(true);
                if (!first_audio_marked_) {
                    // Playout starts once the samples already queued ahead of ours have played.
                    const std::size_t queued = io_->playback->queued();
                    const double ahead = static_cast<double>(queued - std::min(queued, accepted)) / out_rate_;
                    telemetry->mark(utterance_, telemetry::Milestone::FirstAudio, add_seconds(ctx_->now(), ahead));
                    first_audio_marked_ = true;
                }
            }
            written_ += static_cast<std::int64_t>(accepted);
            ready_.erase(ready_.begin(), ready_.begin() + static_cast<std::ptrdiff_t>(accepted));
        }
        break;
    }
    case Mode::Discard:
        written_ += static_cast<std::int64_t>(ready_.size());
        ready_.clear();
        if (!first_audio_marked_) {
            telemetry->mark(utterance_, telemetry::Milestone::FirstAudio, ctx_->now());
            first_audio_marked_ = true;
        }
        break;
    }
    publish_backlog();
    if (utterance_final_ && ready_.empty()) finish_utterance();
}

void PlaybackStage::finish_utterance() {
    utterance_final_ = false;
    if (mode_ == Mode::Live) io_->playback->set_streaming(false);  // the rest of the queue is its tail
    Frame& out = ctx_->make(FrameKind::Playout);
    out.utterance = utterance_;
    out.src_start = src_start_;
    out.src_end = src_end_;
    out.sample_rate = out_rate_;
    out.out_start = mode_ == Mode::Offline ? out_start_ : src_end_;
    out.out_end = out.out_start + static_cast<double>(written_) / out_rate_;
    ctx_->emit(out);
}

void PlaybackStage::tick() {
    if (mode_ != Mode::Live) return;
    if (!ready_.empty()) deliver(utterance_final_);
    if (ready_.empty() && io_->playback->queued() == 0) io_->playback_active.store(false);
    publish_backlog();  // the queue drains as it plays
}

void PlaybackStage::close() {
    if (mode_ == Mode::Live && !ready_.empty()) {
        deliver(true);
        if (!ready_.empty()) log::warn("playback: ", ready_.size(), " samples did not fit the output queue at shutdown");
    }
    if (utterance_final_) finish_utterance();
}

}  // namespace ee
