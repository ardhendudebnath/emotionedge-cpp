#include "core/prosody/controller_stage.hpp"

#include "core/telemetry/telemetry.hpp"

namespace ee {

void EmotionControllerStage::open(StageContext& ctx) {
    ctx_ = &ctx;
    const Params& p = ctx.params();
    controller_ = EmotionController(ControllerConfig::from(p));
    if (const std::string path = p.str("expressivity"); !path.empty()) profiles_ = ExpressivityProfiles::load(path);
    if (const std::string path = p.str("style_anchors"); !path.empty()) styles_ = StyleBank::load(path);
    style_temperature_ = p.real("style_temperature", style_temperature_);
    corrections_ = &ctx.services().telemetry->metrics().counter(
        "ee_controller_corrections_total", "Closed-loop corrections applied after low ECS");
}

void EmotionControllerStage::process(Frame& f) {
    if (f.kind == FrameKind::Feedback) {
        if (controller_.feedback(f.score, f.delta, f.axis_confidence)) corrections_->inc();
        return;
    }
    if (f.kind != FrameKind::Translation || !f.is_final() || f.text.empty()) return;

    controller_.next_utterance();
    const Vad target = controller_.target(f.emotion.vad);
    const ExpressivityProfile profile = profiles_.get(f.language);

    Frame& out = ctx_->make(FrameKind::Speech);
    out.copy_header_from(f);
    out.flags = frame_flags::kFinal;
    out.text = f.text;
    out.language = f.language;
    out.emphasis = f.emphasis;
    out.emotion = f.emotion;  // the source emotion, for the consistency check
    out.axis_confidence = f.axis_confidence;
    out.prosody = controller_.plan(target, f.emotion.confidence, profile);
    out.style = styles_.blend(target, style_temperature_);
    out.delta = target - f.emotion.vad;  // the correction in effect, for the session log
    ctx_->services().telemetry->mark(f.utterance, telemetry::Milestone::ControllerDone, ctx_->now());
    ctx_->emit(out);
}

}  // namespace ee
