#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "core/runtime/frame.hpp"
#include "core/runtime/params.hpp"

namespace ee {

enum class ThreadPriority { Low, Normal, High, Realtime };

/// One scheduler thread (blueprint p.2 threading model: T1 DSP+VAD, T2 ASR, ...).
struct ThreadSpec {
    std::string name;
    /// CPUs to pin to; empty = any. Worker threads that an engine starts from this thread inherit
    /// the set (whisper.cpp starts its workers on every decode), so it must cover the stage's
    /// `threads`; validate() enforces that.
    std::vector<int> cores;
    ThreadPriority priority = ThreadPriority::Normal;
};

/// One stage instance in the pipeline graph.
struct StageSpec {
    std::string name;    ///< unique instance name, referenced by edges
    std::string type;    ///< StageRegistry type, e.g. "segmenter"
    std::string thread;  ///< ThreadSpec::name that runs it
    int tick_ms = 0;     ///< periodic tick() interval; 0 = none
    Params params;
};

/// A lock-free SPSC queue from one stage to another, optionally filtered by frame kind.
struct EdgeSpec {
    std::string from;
    std::string to;
    std::uint32_t kinds = kAllKinds;  ///< bitmask of kind_bit(FrameKind)
    std::size_t capacity = 64;
    /// Feedback edges close loops (5.2 -> 4.1). They never block the producer and do not
    /// count toward end-of-stream, so a loop cannot deadlock the pipeline.
    bool feedback = false;
};

struct TelemetrySpec {
    int flush_ms = 1000;          ///< T7 flush period in threaded mode
    std::string prometheus_path;  ///< text-exposition file rewritten on every flush
    std::string trace_path;       ///< Chrome/Perfetto JSON trace of every process() call
};

/// The pipeline graph in YAML (blueprint "Config & Plugins").
struct PipelineSpec {
    std::string name = "emotionedge";
    std::string source_language = "en";
    std::string target_language = "hi";
    int sample_rate = 16000;
    std::string models_manifest;
    TelemetrySpec telemetry;
    std::vector<ThreadSpec> threads;
    std::vector<StageSpec> stages;
    std::vector<EdgeSpec> edges;

    [[nodiscard]] const StageSpec* find_stage(std::string_view name) const;
    [[nodiscard]] StageSpec* find_stage(std::string_view name);
};

/// Parses a pipeline YAML file. `${config_dir}` in any string value expands to the
/// directory containing the file. Throws ConfigError; the result is validated.
[[nodiscard]] PipelineSpec load_pipeline(const std::filesystem::path& yaml_path);
[[nodiscard]] PipelineSpec parse_pipeline(std::string_view yaml_text,
                                          const std::filesystem::path& config_dir = {});

/// Applies a `key=value` override: `pipeline.target_language=es` or `<stage>.<param>=<value>`
/// (e.g. `asr.engine=whisper`, `asr.tick_ms=10`).
void apply_override(PipelineSpec& spec, std::string_view key, std::string_view value);

/// Throws ConfigError on duplicate names, unknown threads/stages in edges, or a cycle that
/// is not broken by a feedback edge.
void validate(const PipelineSpec& spec);

/// Stage indices ordered so every non-feedback edge goes from an earlier to a later stage.
[[nodiscard]] std::vector<std::size_t> topological_order(const PipelineSpec& spec);

[[nodiscard]] ThreadPriority parse_priority(std::string_view name);

}  // namespace ee
