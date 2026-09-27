#include "core/emotion/consistency_stage.hpp"

#include "core/emotion/prosody_features.hpp"
#include "core/telemetry/ecs.hpp"
#include "core/telemetry/telemetry.hpp"

namespace ee {

void EmotionConsistencyStage::open(StageContext& ctx) {
    ctx_ = &ctx;
    const Params& p = ctx.params();
    rate_ = ctx.pipeline().sample_rate;
    acoustic_ = make_acoustic_model(p, ctx.services().models);  // the same engine as stage 2.2
    lexical_ = make_lexical_model(p, ctx.services().models);
    fusion_ = FusionConfig::from(p);
    threshold_ = p.real("threshold", threshold_);
    below_threshold_ = &ctx.services().telemetry->metrics().counter(
        "ee_ecs_below_threshold_total", "Clauses whose emotion consistency fell below the threshold");
}

void EmotionConsistencyStage::process(Frame& f) {
    if (f.kind != FrameKind::SynthAudio || f.sample_rate <= 0) return;
    if (!resampler_ || resampler_in_rate_ != f.sample_rate) {
        resampler_in_rate_ = f.sample_rate;
        resampler_ = std::make_unique<Resampler>(f.sample_rate, rate_);
    }

    if (f.has(frame_flags::kCalibration)) {
        // The TTS voice's neutral render sets the output-side baseline ("relative to the
        // target-language baseline").
        std::vector<float> audio;
        resampler_->reset();
        resampler_->process(f.audio, audio);
        resampler_->flush(audio);
        ProsodyTracker tracker(rate_);
        tracker.push(audio);
        tracker.flush();
        acoustic_->calibrate(summarize(tracker.frames(), tracker.hop_seconds()));
        resampler_->reset();
        return;
    }

    if (f.utterance != utterance_) {
        utterance_ = f.utterance;
        clause_index_ = 0;
        clause_.clear();
        resampler_->reset();
    }
    if (clause_.empty()) clause_text_ = f.text;
    resampler_->process(f.audio, clause_);
    if (f.has(frame_flags::kClauseEnd) || f.is_final()) {
        resampler_->flush(clause_);
        evaluate_clause(f);
        clause_.clear();
        resampler_->reset();
        ++clause_index_;
    }
}

void EmotionConsistencyStage::evaluate_clause(const Frame& chunk) {
    ProsodyTracker tracker(rate_);
    tracker.push(clause_);
    tracker.flush();
    const ModalityEstimate acoustic = acoustic_->estimate(clause_, tracker.frames(), tracker.hop_seconds());
    ModalityEstimate lexical;
    if (lexical_ != nullptr && lexical_->supports(chunk.language)) lexical = lexical_->estimate(clause_text_, chunk.language);
    Vad axes;
    const EmotionState out = fuse(acoustic, lexical, fusion_, &axes);
    if (!acoustic.valid && !lexical.valid) return;  // nothing measurable (e.g. a silent clause)

    const Vad src = chunk.emotion.vad;
    const float ecs = emotion_consistency(src, out.vad);
    last_score_ = ecs;
    ctx_->services().telemetry->record_ecs(ecs);
    if (ecs < threshold_) below_threshold_->inc();

    Frame& fb = ctx_->make(FrameKind::Feedback);
    fb.copy_header_from(chunk);
    fb.utterance = utterance_;
    fb.seq = clause_index_;
    fb.score = ecs;
    fb.delta = out.vad - src;
    fb.emotion = out;
    fb.axis_confidence = axes;
    fb.text = clause_text_;
    fb.language = chunk.language;
    ctx_->emit(fb);
}

}  // namespace ee
