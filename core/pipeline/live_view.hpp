#pragma once

#include <atomic>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include "core/runtime/stage.hpp"

namespace ee {

/// What a live UI shows of a running session (5.3), built from the recorder's events
/// (event_json): per utterance, the source and its translation (partial or draft until final),
/// the emotion read from the speaker, ECS and what 5.2 heard on each synthesized clause, and
/// where it played. Events arrive on pipeline threads; a UI reads snapshots.
class LiveView final : public IEventListener {
public:
    struct Utterance {
        std::uint64_t id = 0;
        std::string source;
        bool source_final = false;
        std::string translation;
        bool translation_final = false;
        std::string emotion;  ///< label; empty until the state tracker (3.1) has spoken
        float valence = 0.0f;
        float arousal = 0.0f;
        float dominance = 0.0f;
        float confidence = 0.0f;
        std::vector<std::string> emphasis;  ///< emphasized source words
        std::vector<float> ecs;             ///< one per synthesized clause
        std::vector<std::string> heard;     ///< the emotion 5.2 heard on each clause
        double src_start = 0.0;
        double src_end = 0.0;
        double out_start = -1.0;  ///< output timeline, -1 until played
        double out_end = -1.0;
    };

    void on_event(std::string_view json) override;

    /// The utterances so far, in order.
    [[nodiscard]] std::vector<Utterance> snapshot() const;
    /// Bumps with every event, so a UI can tell whether anything changed.
    [[nodiscard]] std::uint64_t version() const noexcept { return version_.load(std::memory_order_acquire); }
    void clear();

private:
    mutable std::mutex mu_;
    std::map<std::uint64_t, Utterance> utterances_;
    std::atomic<std::uint64_t> version_{0};
};

}  // namespace ee
