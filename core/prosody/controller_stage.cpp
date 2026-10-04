#include "core/prosody/controller_stage.hpp"

#include <filesystem>

#include "core/runtime/log.hpp"
#include "core/runtime/model_registry.hpp"
#include "core/telemetry/telemetry.hpp"

namespace ee {

void EmotionControllerStage::open(StageContext& ctx) {
    ctx_ = &ctx;
    const Params& p = ctx.params();
    controller_ = EmotionController(ControllerConfig::from(p));
    // A learned plan travels with the TTS voice it was learned on (plan_model / plan_model_id,
    // e.g. the Kokoro voice directory, holding plan_file). learned_plan: false keeps the rules;
    // plan_emotions limits it to the emotions that also held up end to end.
    if (const std::string dir = p.flag("learned_plan", true) ? resolve_model_path(p, ctx.services().models, "plan_model")
                                                              : std::string();
        !dir.empty()) {
        const std::filesystem::path file = std::filesystem::path(dir) / p.str("plan_file", "prosody_controller.json");
        if (std::filesystem::exists(file)) {
            LearnedProsody learned = LearnedProsody::load(file, p.list("plan_emotions"));
            log::info("controller: learned prosody plan for ", learned.controls.size(), " emotion(s) from ", file.string());
            controller_.set_learned(std::move(learned));
        } else {
            log::warn("controller: no learned prosody plan at ", file.string(), "; using the rules");
        }
    }
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
