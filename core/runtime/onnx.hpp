#pragma once

// Inference Runtime (blueprint "Edge runtime"): the ONNX Runtime environment shared by every
// ORT-backed engine, with execution providers chosen from a device profile.
#if !defined(EE_HAVE_ONNXRUNTIME)
#error "core/runtime/onnx.hpp needs a build with -DEE_WITH_ONNXRUNTIME=ON"
#endif

#include <onnxruntime_cxx_api.h>

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "core/runtime/model_registry.hpp"
#include "core/runtime/params.hpp"

namespace ee::onnx {

struct SessionConfig {
    std::vector<std::string> providers{"cpu"};  ///< cpu | cuda | tensorrt | openvino | coreml | xnnpack
    int intra_threads = 1;
    int inter_threads = 1;
    /// Let idle intra-op workers spin between runs. Spinning shaves a little latency off each run
    /// but burns cores that whisper and CTranslate2 need, so it is off unless a stage asks.
    bool spin = false;
    int gpu_id = 0;
    /// Non-empty: ORT writes a per-node profile (op, provider, time) to `<profile>_<date>.json`,
    /// e.g. to see which nodes of a GPU session run on the CPU.
    std::string profile;
};

/// Reads `device` (a manifest device profile such as "cpu"/"cuda", or "auto": cuda when this
/// ORT build has it), `providers`, `threads`, `spin`, `gpu_id` and `ort_profile` from stage
/// params; explicit params override the manifest's device profile.
/// With a `prefix` (e.g. "acoustic_"), `<prefix>device` etc. come first, so one stage can run
/// its models on different devices.
[[nodiscard]] SessionConfig session_config(const Params& params, const ModelRegistry* registry,
                                           std::string_view prefix = {});

/// The execution providers this ONNX Runtime build has, in ORT's names
/// ("CUDAExecutionProvider", ...).
[[nodiscard]] const std::vector<std::string>& available_providers();

/// Whether this build can run a provider named as in SessionConfig ("cuda", "cpu", ...).
[[nodiscard]] bool provider_available(std::string_view name);

/// The process-wide ORT environment.
[[nodiscard]] Ort::Env& env();

/// Loads a model with the requested providers in priority order. Providers this ORT build (or
/// machine) lacks are skipped with a warning; the CPU provider is always the final fallback.
[[nodiscard]] Ort::Session load_session(const std::string& path, const SessionConfig& config);

/// One session per (model, config) for the whole process, e.g. emotion2vec+ used by both the
/// emotion engine (2.2) and the consistency check (5.2), or by every session of the streaming
/// server. ORT sessions are safe to run from several threads. The session lives while any caller
/// holds it. Every ORT engine loads its model this way.
[[nodiscard]] std::shared_ptr<Ort::Session> shared_session(const std::string& path, const SessionConfig& config);

}  // namespace ee::onnx
