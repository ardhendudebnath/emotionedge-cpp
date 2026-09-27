#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

#include "core/runtime/clock.hpp"
#include "core/telemetry/metrics.hpp"

namespace ee::telemetry {

/// Points in an utterance's life, from the speaker falling silent to translated audio playing.
enum class Milestone : std::uint8_t {
    SpeechEnd,       ///< last voiced audio of the utterance was captured
    Endpoint,        ///< segmenter declared the endpoint (after the hangover)
    AsrFinal,        ///< final transcript emitted
    EmotionFinal,    ///< utterance-level acoustic emotion fused (parallel with ASR)
    StateReady,      ///< text + emotion joined by the state tracker
    MtFinal,         ///< final translation emitted
    ControllerDone,  ///< prosody plan and style vector emitted
    TtsFirstChunk,   ///< first synthesized audio chunk emitted
    FirstAudio,      ///< first translated sample reached the speaker (or output timeline)
};
inline constexpr std::size_t kMilestoneCount = 9;

[[nodiscard]] std::string_view to_string(Milestone m) noexcept;

/// One row of the latency budget (blueprint p.2). The row spans from the latest of the
/// `from` milestones to `to`.
struct BudgetRow {
    std::string_view name;
    std::uint32_t from_mask;
    Milestone to;
    std::uint64_t target_us;
    bool on_critical_path;  ///< counts toward the 735 ms total (emotion fusion runs in parallel)
};

/// The eight budget rows, in blueprint order.
[[nodiscard]] const std::array<BudgetRow, 8>& latency_budget() noexcept;

/// End-to-end target: speech end -> first translated audio, p95.
inline constexpr std::uint64_t kEndToEndTargetUs = 800'000;

/// Cross-cutting telemetry shared by every stage: metrics, the per-utterance latency timeline,
/// real-time factor and emotion consistency. All recording paths are lock-free.
class Telemetry {
public:
    explicit Telemetry(std::size_t timeline_slots = 256);
    Telemetry(const Telemetry&) = delete;
    Telemetry& operator=(const Telemetry&) = delete;

    [[nodiscard]] MetricsRegistry& metrics() noexcept { return metrics_; }
    [[nodiscard]] const MetricsRegistry& metrics() const noexcept { return metrics_; }

    /// Opens the timeline slot for a new utterance (the segmenter calls this at speech start).
    void begin_utterance(std::uint64_t utterance) noexcept;
    /// Records a milestone; the first mark wins. Marking FirstAudio completes the utterance and
    /// folds its budget rows into the latency histograms.
    void mark(std::uint64_t utterance, Milestone m, TimePoint t = Clock::now()) noexcept;
    /// Microseconds since this Telemetry was created, or -1 when the milestone is unknown.
    [[nodiscard]] std::int64_t milestone_us(std::uint64_t utterance, Milestone m) const noexcept;

    void record_ecs(float ecs) noexcept;
    void record_rtf(float rtf) noexcept;
    void count_dropout(std::uint64_t samples) noexcept;

    struct RowReport {
        const BudgetRow* row;
        Histogram::Snapshot snapshot;
    };
    [[nodiscard]] std::vector<RowReport> budget_report() const;
    [[nodiscard]] Histogram::Snapshot end_to_end() const { return end_to_end_->snapshot(); }
    [[nodiscard]] Histogram::Snapshot ecs() const { return ecs_->snapshot(); }
    [[nodiscard]] Histogram::Snapshot rtf() const { return rtf_->snapshot(); }
    [[nodiscard]] std::uint64_t completed_utterances() const noexcept { return completed_->value(); }
    [[nodiscard]] std::uint64_t dropouts() const noexcept { return dropouts_->value(); }

private:
    static constexpr std::int64_t kUnset = INT64_MIN;
    struct Slot {
        std::atomic<std::uint64_t> utterance{0};
        std::array<std::atomic<std::int64_t>, kMilestoneCount> t{};
        std::atomic<bool> done{false};
    };
    void complete(Slot& slot) noexcept;

    MetricsRegistry metrics_;
    TimePoint epoch_;
    std::size_t slot_count_;
    std::unique_ptr<Slot[]> slots_;
    std::array<Histogram*, 8> rows_{};
    Histogram* end_to_end_ = nullptr;
    Histogram* ecs_ = nullptr;
    Histogram* rtf_ = nullptr;
    Counter* completed_ = nullptr;
    Counter* dropouts_ = nullptr;
};

}  // namespace ee::telemetry
