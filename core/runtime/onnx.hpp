#pragma once

// Inference Runtime (blueprint "Edge runtime"): the ONNX Runtime environment shared by every
// ORT-backed engine, with execution providers chosen from a device profile.
#if !defined(EE_HAVE_ONNXRUNTIME)
#error "core/runtime/onnx.hpp needs a build with -DEE_WITH_ONNXRUNTIME=ON"
#endif

#include <onnxruntime_cxx_api.h>

#include <memory>
#include <string>
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
};

/// Reads `device` (a manifest device profile such as "cpu"/"cuda"), `providers`, `threads` and
/// `spin` from stage params; explicit params override the profile.
[[nodiscard]] SessionConfig session_config(const Params& params, const ModelRegistry* registry);

/// The process-wide ORT environment.
[[nodiscard]] Ort::Env& env();

/// Loads a model with the requested providers in priority order. Providers this ORT build lacks
/// are skipped with a warning; the CPU provider is always the final fallback.
[[nodiscard]] Ort::Session load_session(const std::string& path, const SessionConfig& config);

/// One session per (model, config) for the whole process, e.g. emotion2vec+ used by both the
/// emotion engine (2.2) and the consistency check (5.2). ORT sessions are safe to run from
/// several threads. The session lives while any caller holds it.
[[nodiscard]] std::shared_ptr<Ort::Session> shared_session(const std::string& path, const SessionConfig& config);

}  // namespace ee::onnx
