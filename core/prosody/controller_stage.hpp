#pragma once

#include <string>

#include "core/prosody/controller.hpp"
#include "core/prosody/expressivity.hpp"
#include "core/prosody/style.hpp"
#include "core/runtime/stage.hpp"
#include "core/telemetry/metrics.hpp"

namespace ee {

/// Stage 4.1: turns each final translation into a Speech request (text + prosody plan + style
/// vector), relative to the target language's baseline, and applies ECS feedback from 5.2 to the
/// next plan. The Speech frame keeps the *source* emotion so 5.2 can score against it.
class EmotionControllerStage final : public IStage {
public:
    void open(StageContext& ctx) override;
    void process(Frame& frame) override;

    [[nodiscard]] const EmotionController& controller() const noexcept { return controller_; }

private:
    StageContext* ctx_ = nullptr;
    EmotionController controller_;
    ExpressivityProfiles profiles_ = ExpressivityProfiles::defaults();
    StyleBank styles_ = StyleBank::placeholder();
    float style_temperature_ = 0.35f;
    telemetry::Counter* corrections_ = nullptr;
};

}  // namespace ee
