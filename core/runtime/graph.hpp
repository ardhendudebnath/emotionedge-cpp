#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <string_view>

#include "core/runtime/config.hpp"
#include "core/runtime/stage.hpp"
#include "core/runtime/stage_registry.hpp"

namespace ee {

enum class ExecutionMode { Deterministic, Threaded };

/// Runs a PipelineSpec (blueprint "Scheduler"): every edge is a bounded lock-free SPSC queue
/// of pre-allocated frames, and stages run either
///  - deterministically on the calling thread, in topological order (offline runs, tests), or
///  - on one thread per ThreadSpec, core-pinned and prioritized (live runs).
///
/// Backpressure: when an edge is full, non-final hypotheses (and feedback) are dropped, while
/// reliable frames (audio, finals) wait up to 2 ms and are then parked in an overflow the
/// consumer drains in order, so stale partials are shed, audio never is, and a full queue can
/// never deadlock the scheduler, even when stages share a thread or threads form a cycle.
/// A consumer also skips a partial that a newer hypothesis for the same utterance superseded.
/// End-of-stream flows along non-feedback edges; each stage closes once all such inputs ended.
class Graph {
public:
    Graph(PipelineSpec spec, const StageRegistry& registry, Services services);
    ~Graph();
    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;

    /// Opens every stage and runs them on the calling thread until all have closed.
    /// Returns false if the pipeline stalled with stages still open.
    bool run_deterministic();

    /// Opens every stage on the calling thread (so errors surface here), then starts the
    /// scheduler threads and, if configured, the telemetry flush thread (T7).
    void start();
    /// Waits until every stage has closed (true) or the timeout expires (false).
    bool wait(std::chrono::milliseconds timeout = std::chrono::milliseconds::max());
    /// Stops the threads without draining and joins them. Idempotent.
    void stop();

    [[nodiscard]] bool finished() const noexcept;
    [[nodiscard]] const PipelineSpec& spec() const noexcept;
    [[nodiscard]] Services& services() noexcept;
    /// The stage instance with this name, or nullptr.
    [[nodiscard]] IStage* stage(std::string_view name) noexcept;
    /// Frames shed under backpressure, and partials skipped because a newer one was queued.
    [[nodiscard]] std::uint64_t dropped_frames() const noexcept;
    [[nodiscard]] std::uint64_t stale_frames() const noexcept;

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

}  // namespace ee
