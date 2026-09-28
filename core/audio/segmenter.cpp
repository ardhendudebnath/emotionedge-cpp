#include "core/audio/segmenter.hpp"

#include <algorithm>
#include <cmath>

#include "core/audio/audio_io.hpp"
#include "core/telemetry/telemetry.hpp"

namespace ee {

// ---- EndpointTracker ----------------------------------------------------------------------------

EndpointTracker::EndpointTracker(SegmenterConfig config, double window_seconds) : cfg_(config) {
    const auto windows = [&](double ms) {
        return std::max(1, static_cast<int>(std::ceil(ms / 1000.0 / window_seconds - 1e-9)));
    };
    min_speech_windows_ = windows(cfg_.min_speech_ms);
    hangover_windows_ = windows(cfg_.hangover_ms);
    max_windows_ = std::max(1, static_cast<int>(cfg_.max_utterance_s / window_seconds));
}

void EndpointTracker::reset() {
    in_speech_ = false;
    onset_run_ = 0;
    quiet_run_ = 0;
    utterance_windows_ = 0;
}

EndpointTracker::Action EndpointTracker::update(float probability) {
    if (!in_speech_) {
        if (probability >= cfg_.threshold_on) {
            if (++onset_run_ >= min_speech_windows_) {
                in_speech_ = true;
                onset_run_ = 0;
                quiet_run_ = 0;
                utterance_windows_ = min_speech_windows_;
                return Action::Start;
            }
        } else {
            onset_run_ = 0;
        }
        return Action::Silence;
    }
    ++utterance_windows_;
    quiet_run_ = probability >= cfg_.threshold_off ? 0 : quiet_run_ + 1;
    if (quiet_run_ >= hangover_windows_ || utterance_windows_ >= max_windows_) {
        in_speech_ = false;
        onset_run_ = 0;
        return Action::End;
    }
    return Action::Continue;
}

// ---- SegmenterStage -----------------------------------------------------------------------------

void SegmenterStage::open(StageContext& ctx) {
    ctx_ = &ctx;
    const Params& p = ctx.params();
    rate_ = ctx.pipeline().sample_rate;
    detector_ = make_speech_detector(p, ctx.services().models, rate_);
    window_ = detector_->window();

    SegmenterConfig cfg;
    cfg.threshold_on = p.real("threshold_on", cfg.threshold_on);
    cfg.threshold_off = p.real("threshold_off", cfg.threshold_off);
    cfg.min_speech_ms = static_cast<int>(p.integer("min_speech_ms", cfg.min_speech_ms));
    cfg.hangover_ms = static_cast<int>(p.integer("hangover_ms", cfg.hangover_ms));
    cfg.preroll_ms = static_cast<int>(p.integer("preroll_ms", cfg.preroll_ms));
    cfg.max_utterance_s = p.real("max_utterance_s", cfg.max_utterance_s);
    if (cfg.threshold_off > cfg.threshold_on) throw ConfigError("segmenter: threshold_off must not exceed threshold_on");
    // Barge-in cancels the translation still playing when speech starts. That suits a listener
    // talking over the playback. For one speaker who keeps talking (interpreting a monologue), it
    // drops translations, so it can be turned off; the translations then queue.
    barge_in_ = p.flag("barge_in", true);
    tracker_ = std::make_unique<EndpointTracker>(cfg, static_cast<double>(window_) / rate_);
    preroll_windows_ = std::max<std::size_t>(
        1, static_cast<std::size_t>(cfg.preroll_ms) * static_cast<std::size_t>(rate_) / 1000 / window_);
    pending_.reserve(window_ * 4);
    current_.samples.reserve(window_);
}

TimePoint SegmenterStage::origin_at(std::int64_t pos) const {
    // Offline runs process audio faster than real time: mixing in audio-time offsets would make
    // latencies meaningless, so there every event is stamped with its processing time.
    if (ctx_->deterministic()) return last_frame_origin_;
    return add_seconds(last_frame_origin_, static_cast<double>(pos - last_frame_pos_) / rate_);
}

void SegmenterStage::process(Frame& in) {
    if (in.kind != FrameKind::Audio || in.audio.empty()) return;
    if (!have_origin_) {
        pending_pos_ = in.stream_pos;
        have_origin_ = true;
    }
    last_frame_pos_ = in.stream_pos;
    last_frame_origin_ = in.t_origin;
    pending_.insert(pending_.end(), in.audio.begin(), in.audio.end());

    std::size_t offset = 0;
    while (pending_.size() - offset >= window_) {
        current_.pos = pending_pos_;
        current_.origin = origin_at(pending_pos_);
        current_.samples.assign(pending_.begin() + static_cast<std::ptrdiff_t>(offset),
                                pending_.begin() + static_cast<std::ptrdiff_t>(offset + window_));
        handle_window(detector_->probability(current_.samples));
        offset += window_;
        pending_pos_ += static_cast<std::int64_t>(window_);
    }
    pending_.erase(pending_.begin(), pending_.begin() + static_cast<std::ptrdiff_t>(offset));
}

void SegmenterStage::handle_window(float probability) {
    switch (tracker_->update(probability)) {
    case EndpointTracker::Action::Silence: {
        // Keep recent silence as pre-roll, recycling window buffers.
        Window w;
        if (!spare_.empty()) {
            w = std::move(spare_.back());
            spare_.pop_back();
        }
        w.pos = current_.pos;
        w.origin = current_.origin;
        w.samples.assign(current_.samples.begin(), current_.samples.end());
        preroll_.push_back(std::move(w));
        while (preroll_.size() > preroll_windows_) {
            spare_.push_back(std::move(preroll_.front()));
            preroll_.pop_front();
        }
        break;
    }
    case EndpointTracker::Action::Start: {
        ++utterance_;
        seq_ = 0;
        ctx_->services().telemetry->begin_utterance(utterance_);
        utterance_start_pos_ = preroll_.empty() ? current_.pos : preroll_.front().pos;
        speech_end_pos_ = current_.pos + static_cast<std::int64_t>(window_);
        if (AudioIo* io = ctx_->services().audio; barge_in_ && io != nullptr && io->playback_active.load()) {
            Frame& barge = ctx_->make(FrameKind::Control);
            barge.flags = frame_flags::kBargeIn;
            barge.utterance = utterance_;
            barge.t_origin = current_.origin;
            ctx_->emit(barge);
        }
        bool first = true;
        for (Window& w : preroll_) {
            emit_window(w, first);
            first = false;
            spare_.push_back(std::move(w));
        }
        preroll_.clear();
        emit_window(current_, first);
        break;
    }
    case EndpointTracker::Action::Continue:
        // A window at or above threshold_off resets the quiet run: speech is still going.
        if (tracker_->trailing_quiet_windows() == 0) {
            speech_end_pos_ = current_.pos + static_cast<std::int64_t>(window_);
        }
        emit_window(current_, false);
        break;
    case EndpointTracker::Action::End:
        emit_window(current_, false);  // the window that completed the hangover
        end_utterance();
        break;
    }
}

void SegmenterStage::emit_window(const Window& w, bool first) {
    Frame& f = ctx_->make(FrameKind::Audio);
    f.utterance = utterance_;
    f.seq = seq_++;
    f.stream_pos = w.pos;
    f.t_origin = w.origin;
    f.sample_rate = rate_;
    f.src_start = static_cast<double>(utterance_start_pos_) / rate_;
    f.audio.assign(w.samples.begin(), w.samples.end());
    if (first) f.flags |= frame_flags::kSpeechStart;
    ctx_->emit(f);
}

void SegmenterStage::end_utterance() {
    const TimePoint speech_end = origin_at(speech_end_pos_);
    auto* telemetry = ctx_->services().telemetry;
    telemetry->mark(utterance_, telemetry::Milestone::SpeechEnd, speech_end);
    telemetry->mark(utterance_, telemetry::Milestone::Endpoint, ctx_->now());

    Frame& f = ctx_->make(FrameKind::Audio);
    f.utterance = utterance_;
    f.seq = seq_++;
    f.flags = frame_flags::kEndpoint | frame_flags::kFinal;
    f.stream_pos = current_.pos + static_cast<std::int64_t>(current_.samples.size());  // end of the last window
    f.t_origin = speech_end;
    f.sample_rate = rate_;
    f.src_start = static_cast<double>(utterance_start_pos_) / rate_;
    f.src_end = static_cast<double>(speech_end_pos_) / rate_;
    ctx_->emit(f);
}

void SegmenterStage::close() {
    if (tracker_ == nullptr || !tracker_->in_speech()) return;
    // End of stream mid-utterance: flush the partial window, then close the utterance.
    if (!pending_.empty()) {
        current_.pos = pending_pos_;
        current_.origin = origin_at(pending_pos_);
        current_.samples.assign(pending_.begin(), pending_.end());
        emit_window(current_, false);
        pending_pos_ += static_cast<std::int64_t>(pending_.size());
        speech_end_pos_ = pending_pos_;
        pending_.clear();
    }
    tracker_->reset();
    end_utterance();
}

}  // namespace ee
