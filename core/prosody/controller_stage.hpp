#pragma once

#include <cstdint>
#include <string>

#include "core/prosody/controller.hpp"
#include "core/prosody/expressivity.hpp"
#include "core/prosody/style.hpp"
#include "core/runtime/stage.hpp"
#include "core/telemetry/metrics.hpp"

namespace ee {

/// Stage 4.1: turns each final translation into a Speech request (text + prosody plan + style
/// vector), relative to the target language's baseline, and applies ECS feedback from 5.2 to the
/// next plan. The Speech frame keeps the *source* emotion so 5.2 can score against it. A
/// translation's leading words (frame_flags::kLeading) are planned as the final would be, and
/// the final that follows them is spoken with the same plan.
class EmotionControllerStage final : public IStage {
public:
    void open(StageContext& ctx) override;
    void process(Frame& frame) override;

    [[nodiscard]] const EmotionController& controller() const noexcept { return controller_; }

private:
    struct Plan {
        ProsodyTargets prosody;
        StyleVector style{};
        Vad delta;
    };

    StageContext* ctx_ = nullptr;
    std::uint64_t planned_utterance_ = 0;  ///< leading words seen, final not yet
    Plan plan_;
    EmotionController controller_;
    ExpressivityProfiles profiles_ = ExpressivityProfiles::defaults();
    StyleBank styles_ = StyleBank::placeholder();
    float style_temperature_ = 0.35f;
    telemetry::Counter* corrections_ = nullptr;
};

}  // namespace ee
