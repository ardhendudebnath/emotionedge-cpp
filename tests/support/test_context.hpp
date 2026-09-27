#pragma once

#include <string>
#include <vector>

#include "core/runtime/stage.hpp"
#include "core/telemetry/telemetry.hpp"

namespace ee::test {

/// A StageContext that records emitted frames, for testing one stage in isolation.
class RecordingContext final : public StageContext {
public:
    explicit RecordingContext(Params params = {}, std::string name = "stage-under-test")
        : name_(std::move(name)), params_(std::move(params)) {
        services_.telemetry = &telemetry_;
    }

    std::string_view name() const noexcept override { return name_; }
    const Params& params() const noexcept override { return params_; }
    const PipelineSpec& pipeline() const noexcept override { return pipeline_; }
    Services& services() noexcept override { return services_; }
    Frame& make(FrameKind kind) override {
        scratch_.reset(kind);
        return scratch_;
    }
    void emit(const Frame& frame) override { emitted.push_back(frame); }
    void finish() override { finished = true; }
    bool deterministic() const noexcept override { return true; }

    Params& mutable_params() { return params_; }
    PipelineSpec& mutable_pipeline() { return pipeline_; }
    telemetry::Telemetry& telemetry() { return telemetry_; }

    /// Emitted frames of one kind.
    std::vector<Frame> of(FrameKind kind) const {
        std::vector<Frame> out;
        for (const Frame& f : emitted) {
            if (f.kind == kind) out.push_back(f);
        }
        return out;
    }

    std::vector<Frame> emitted;
    bool finished = false;

private:
    std::string name_;
    Params params_;
    PipelineSpec pipeline_;
    telemetry::Telemetry telemetry_;
    Services services_;
    Frame scratch_;
};

}  // namespace ee::test
