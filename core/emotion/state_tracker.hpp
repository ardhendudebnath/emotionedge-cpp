#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "core/emotion/fusion.hpp"
#include "core/runtime/stage.hpp"

namespace ee {

/// Stage 3.1 "Context & Emotion State". Joins each utterance's final transcript with its final
/// emotion, stabilizes the label with hysteresis, marks emphasized words (energy peaks aligned
/// to word timestamps) and emits one Utterance frame: `EmotionState{V, A, D, conf}` + text.
/// Live runs stop waiting for the emotion after `join_timeout_ms` and use the latest estimate.
class StateTrackerStage final : public IStage {
public:
    void open(StageContext& ctx) override;
    void process(Frame& frame) override;
    void tick() override;
    void close() override;

private:
    struct Pending {
        std::uint64_t id = 0;
        std::optional<Frame> transcript;
        std::optional<Frame> emotion;
        TimePoint transcript_seen{};
    };
    Pending& pending(std::uint64_t id);
    void try_emit(std::uint64_t id, bool force);

    StageContext* ctx_ = nullptr;
    LabelHysteresis hysteresis_;
    EmphasisConfig emphasis_;
    bool wait_for_emotion_ = true;
    double join_timeout_s_ = 0.15;
    EmotionState latest_;
    Vad latest_axes_;
    std::vector<Pending> pending_;
};

}  // namespace ee
