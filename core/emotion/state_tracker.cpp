#include "core/emotion/state_tracker.hpp"

#include <algorithm>

#include "core/telemetry/telemetry.hpp"

namespace ee {

void StateTrackerStage::open(StageContext& ctx) {
    ctx_ = &ctx;
    const Params& p = ctx.params();
    hysteresis_ = LabelHysteresis(p.real("hysteresis.margin", 0.08f), p.real("hysteresis.strong_margin", 0.25f),
                                  static_cast<int>(p.integer("hysteresis.updates", 2)));
    emphasis_.z_threshold = p.real("emphasis.z", emphasis_.z_threshold);
    emphasis_.min_rise_db = p.real("emphasis.min_rise_db", emphasis_.min_rise_db);
    emphasis_.max_words = static_cast<int>(p.integer("emphasis.max_words", emphasis_.max_words));
    wait_for_emotion_ = p.flag("wait_for_emotion", true);
    join_timeout_s_ = p.number("join_timeout_ms", 150.0) / 1000.0;
}

StateTrackerStage::Pending& StateTrackerStage::pending(std::uint64_t id) {
    for (Pending& p : pending_) {
        if (p.id == id) return p;
    }
    Pending& p = pending_.emplace_back();
    p.id = id;
    return p;
}

void StateTrackerStage::process(Frame& f) {
    if (f.utterance == 0) return;
    if (f.kind == FrameKind::Emotion) {
        latest_ = f.emotion;
        latest_axes_ = f.axis_confidence;
        if (f.is_final()) {
            pending(f.utterance).emotion = f;
            try_emit(f.utterance, false);
        }
    } else if (f.kind == FrameKind::Transcript && f.is_final()) {
        Pending& p = pending(f.utterance);
        p.transcript = f;
        p.transcript_seen = ctx_->now();
        try_emit(f.utterance, false);
    }
}

void StateTrackerStage::try_emit(std::uint64_t id, bool force) {
    const auto it = std::find_if(pending_.begin(), pending_.end(), [id](const Pending& p) { return p.id == id; });
    if (it == pending_.end() || !it->transcript) return;
    if (!it->emotion && wait_for_emotion_ && !force) return;

    const Frame& t = *it->transcript;
    EmotionState state = it->emotion ? it->emotion->emotion : latest_;
    const Vad axes = it->emotion ? it->emotion->axis_confidence : latest_axes_;
    state.label = hysteresis_.update(state.vad);

    Frame& out = ctx_->make(FrameKind::Utterance);
    out.copy_header_from(t);
    out.flags = frame_flags::kFinal;
    out.text = t.text;
    out.language = t.language;
    out.words = t.words;
    out.emotion = state;
    out.axis_confidence = axes;
    if (it->emotion) {
        out.emphasis = find_emphasis(t.words, it->emotion->envelope, it->emotion->envelope_hop, emphasis_);
    }
    ctx_->services().telemetry->mark(id, telemetry::Milestone::StateReady, ctx_->now());
    ctx_->emit(out);
    pending_.erase(it);
}

void StateTrackerStage::tick() {
    if (ctx_->deterministic()) return;
    const TimePoint now = ctx_->now();
    std::vector<std::uint64_t> due;
    for (const Pending& p : pending_) {
        if (p.transcript && !p.emotion && seconds_between(p.transcript_seen, now) > join_timeout_s_) due.push_back(p.id);
    }
    for (std::uint64_t id : due) try_emit(id, true);
}

void StateTrackerStage::close() {
    std::vector<std::uint64_t> ids;
    for (const Pending& p : pending_) ids.push_back(p.id);
    for (std::uint64_t id : ids) try_emit(id, true);
}

}  // namespace ee
