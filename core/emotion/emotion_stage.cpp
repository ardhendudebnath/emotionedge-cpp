#include "core/emotion/emotion_stage.hpp"

#include <algorithm>

#include "core/telemetry/telemetry.hpp"

namespace ee {

void EmotionEngineStage::open(StageContext& ctx) {
    ctx_ = &ctx;
    const Params& p = ctx.params();
    rate_ = ctx.pipeline().sample_rate;
    acoustic_ = make_acoustic_model(p, ctx.services().models);
    lexical_ = make_lexical_model(p, ctx.services().models);
    fusion_ = FusionConfig::from(p);
    smoother_ = EmotionSmoother(p.real("ema_alpha", 0.5f));
    window_frames_ = static_cast<std::size_t>(std::max<std::int64_t>(10, p.integer("window_ms", 1000) / 10));
    hop_frames_ = static_cast<std::size_t>(std::max<std::int64_t>(5, p.integer("hop_ms", 500) / 10));
    final_alpha_ = p.real("final_alpha", 0.8f);
    wait_for_transcript_ = lexical_ != nullptr && p.flag("wait_for_transcript", true);
    transcript_timeout_s_ = p.number("transcript_timeout_ms", 1500.0) / 1000.0;
    max_audio_ = static_cast<std::size_t>(p.number("max_audio_s", 20.0) * rate_);
}

EmotionEngineStage::Utterance* EmotionEngineStage::record(std::uint64_t id) {
    for (Utterance& u : utterances_) {
        if (u.id == id) return &u;
    }
    // Late frames for an utterance that was already finalized (e.g. after a timeout) are ignored.
    if (std::find(finished_.begin(), finished_.end(), id) != finished_.end()) return nullptr;
    // Keep a few utterances open: the next one can start before this one's final transcript.
    while (utterances_.size() >= 4) {
        try_finalize(utterances_.front(), true);
    }
    Utterance& u = utterances_.emplace_back();
    u.id = id;
    u.prosody = ProsodyTracker(rate_);
    u.next_estimate_frame = window_frames_;
    u.header.reset(FrameKind::Emotion);
    u.header.utterance = id;
    return &u;
}

void EmotionEngineStage::process(Frame& f) {
    if (f.kind == FrameKind::Audio) {
        on_audio(f);
    } else if (f.kind == FrameKind::Transcript) {
        on_transcript(f);
    }
}

void EmotionEngineStage::on_audio(Frame& f) {
    if (f.utterance == 0) return;
    Utterance* rec = record(f.utterance);
    if (rec == nullptr) return;
    Utterance& u = *rec;
    const float hop_s = u.prosody.hop_seconds();
    const std::size_t hop_samples = static_cast<std::size_t>(rate_) / 100;  // 10 ms prosody hop
    if (f.has(frame_flags::kSpeechStart)) {
        u.header.src_start = f.src_start;
        u.header.stream_pos = f.stream_pos;
    }
    if (!f.audio.empty() && !u.endpoint) {
        u.prosody.push(f.audio);
        const std::size_t room = max_audio_ - std::min(max_audio_, u.audio.size());
        u.audio.insert(u.audio.end(), f.audio.begin(),
                       f.audio.begin() + static_cast<std::ptrdiff_t>(std::min(room, f.audio.size())));

        // Partial estimates over the last window while the speaker is still talking.
        const auto& frames = u.prosody.frames();
        while (frames.size() >= u.next_estimate_frame) {
            const std::size_t end = u.next_estimate_frame;
            const std::size_t begin = end - std::min(end, window_frames_);
            const std::size_t audio_end = std::min(u.audio.size(), end * hop_samples);
            const std::size_t audio_begin = audio_end - std::min(audio_end, window_frames_ * hop_samples);
            const ModalityEstimate a = acoustic_->estimate(
                std::span<const float>(u.audio.data() + audio_begin, audio_end - audio_begin),
                std::span<const ProsodyFrame>(frames.data() + begin, end - begin), hop_s);
            if (a.valid || u.lexical.valid) {
                Vad axes;
                const EmotionState fused = fuse(a, u.lexical, fusion_, &axes);
                emit_state(u, smoother_.update(fused), axes, false);
            }
            u.next_estimate_frame += hop_frames_;
        }
    }
    if (f.has(frame_flags::kEndpoint)) {
        u.endpoint = true;
        u.endpoint_seen = ctx_->now();
        u.header.src_start = f.src_start;
        u.header.src_end = f.src_end;
        u.header.t_origin = f.t_origin;
        u.prosody.flush();
        const auto& frames = u.prosody.frames();
        u.acoustic_final = acoustic_->estimate(u.audio, frames, hop_s);
        acoustic_->end_utterance(summarize(frames, hop_s));  // adapt to the speaker afterwards
        ctx_->services().telemetry->mark(u.id, telemetry::Milestone::EmotionFinal, ctx_->now());
        try_finalize(u, false);
    }
}

void EmotionEngineStage::on_transcript(const Frame& f) {
    if (f.utterance == 0) return;
    Utterance* rec = record(f.utterance);
    if (rec == nullptr) return;
    Utterance& u = *rec;
    if (lexical_ != nullptr && lexical_->supports(f.language)) u.lexical = lexical_->estimate(f.text, f.language);
    if (f.is_final()) {
        u.final_text = true;
        try_finalize(u, false);
    }
}

void EmotionEngineStage::try_finalize(Utterance& u, bool force) {
    if (!force && (!u.endpoint || (wait_for_transcript_ && !u.final_text))) return;
    const float hop_s = u.prosody.hop_seconds();
    if (!u.endpoint) {  // forced before the endpoint arrived (end of stream)
        u.prosody.flush();
        u.acoustic_final = acoustic_->estimate(u.audio, u.prosody.frames(), hop_s);
    }
    Vad axes;
    const EmotionState fused = fuse(u.acoustic_final, u.lexical, fusion_, &axes);
    emit_state(u, smoother_.update(fused, final_alpha_), axes, true);
    const std::uint64_t id = u.id;
    finished_.push_back(id);
    if (finished_.size() > 16) finished_.pop_front();
    std::erase_if(utterances_, [id](const Utterance& x) { return x.id == id; });
}

void EmotionEngineStage::emit_state(const Utterance& u, const EmotionState& state, const Vad& axes, bool final) {
    Frame& out = ctx_->make(FrameKind::Emotion);
    out.copy_header_from(u.header);
    out.utterance = u.id;
    out.flags = final ? frame_flags::kFinal : 0u;
    out.emotion = state;
    out.axis_confidence = axes;
    if (final) {
        out.envelope_hop = u.prosody.hop_seconds();
        out.envelope.reserve(u.prosody.frames().size());
        for (const ProsodyFrame& pf : u.prosody.frames()) out.envelope.push_back(pf.energy_db);
    }
    ctx_->emit(out);
}

void EmotionEngineStage::tick() {
    if (ctx_->deterministic()) return;  // no wall-clock timeouts in offline runs
    const TimePoint now = ctx_->now();
    for (std::size_t i = 0; i < utterances_.size();) {
        Utterance& u = utterances_[i];
        if (u.endpoint && !u.final_text && seconds_between(u.endpoint_seen, now) > transcript_timeout_s_) {
            try_finalize(u, true);  // erases u
        } else {
            ++i;
        }
    }
}

void EmotionEngineStage::close() {
    while (!utterances_.empty()) try_finalize(utterances_.front(), true);
}

}  // namespace ee
