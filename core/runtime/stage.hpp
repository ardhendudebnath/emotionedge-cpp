#pragma once

#include <string_view>

#include "core/runtime/clock.hpp"
#include "core/runtime/config.hpp"
#include "core/runtime/frame.hpp"
#include "core/runtime/params.hpp"

namespace ee {

namespace telemetry {
class Telemetry;
}
class ModelRegistry;
struct AudioIo;

/// Receives the pipeline's results as they happen (5.3 live outputs, e.g. a streaming server).
/// Called from stage threads: implementations must be thread-safe and must not block.
class IEventListener {
public:
    virtual ~IEventListener() = default;
    /// One JSON object per event, with a "type" field (see RecorderStage).
    virtual void on_event(std::string_view json) = 0;
};

/// Cross-cutting services shared by every stage (blueprint "Edge runtime").
struct Services {
    telemetry::Telemetry* telemetry = nullptr;  ///< always set by the Graph
    const ModelRegistry* models = nullptr;      ///< optional: models/manifest.json
    AudioIo* audio = nullptr;                   ///< optional: capture/playback endpoints
    IEventListener* events = nullptr;           ///< optional: live results (5.3)
};

/// What a running stage sees of the pipeline: its parameters, the shared services and the
/// output edges it emits into. Implemented by the Graph (and by a recording double in tests).
class StageContext {
public:
    virtual ~StageContext() = default;

    [[nodiscard]] virtual std::string_view name() const noexcept = 0;
    [[nodiscard]] virtual const Params& params() const noexcept = 0;
    [[nodiscard]] virtual const PipelineSpec& pipeline() const noexcept = 0;
    [[nodiscard]] virtual Services& services() noexcept = 0;

    /// Scratch frame reset to `kind`. Owned by the context and reused across calls so its
    /// buffers keep their capacity; build one frame at a time and emit it before the next make().
    [[nodiscard]] virtual Frame& make(FrameKind kind) = 0;
    /// Copies `frame` into every output edge whose kind filter accepts it.
    virtual void emit(const Frame& frame) = 0;
    /// For source stages (no inputs): nothing more will be produced; the stage is then closed.
    virtual void finish() = 0;
    /// True in single-threaded offline runs, where stages must not wait on wall-clock time.
    [[nodiscard]] virtual bool deterministic() const noexcept = 0;
    [[nodiscard]] virtual TimePoint now() const noexcept { return Clock::now(); }
};

/// A pipeline stage. The blueprint's contract is
///     struct IStage { virtual void process(Frame&) = 0; };
/// open/tick/close are optional lifecycle hooks around it.
struct IStage {
    virtual ~IStage() = default;

    /// Called once, before any frame, with the context to emit through. Load models and
    /// validate parameters here; throwing ConfigError aborts pipeline start-up.
    virtual void open(StageContext& ctx) { (void)ctx; }
    /// Handles one input frame. Must not block; results go out through StageContext::emit.
    virtual void process(Frame& frame) = 0;
    /// Periodic work every tick_ms (every scheduler pass in deterministic runs).
    virtual void tick() {}
    /// Called once after every input delivered end-of-stream; flush pending output here.
    virtual void close() {}
};

}  // namespace ee
