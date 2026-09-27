#include "core/runtime/onnx.hpp"

#include <filesystem>
#include <memory>

#include "core/runtime/log.hpp"

namespace ee::onnx {

SessionConfig session_config(const Params& params, const ModelRegistry* registry) {
    SessionConfig cfg;
    if (registry != nullptr) {
        if (const DeviceProfile* profile = registry->device_profile(params.str("device", "cpu"))) {
            if (!profile->execution_providers.empty()) cfg.providers = profile->execution_providers;
            if (profile->threads > 0) cfg.intra_threads = profile->threads;
        }
    }
    if (const auto providers = params.list("providers"); !providers.empty()) cfg.providers = providers;
    cfg.intra_threads = static_cast<int>(params.integer("threads", cfg.intra_threads));
    return cfg;
}

Ort::Env& env() {
    static Ort::Env instance(ORT_LOGGING_LEVEL_WARNING, "emotionedge");
    return instance;
}

namespace {

void append_provider(Ort::SessionOptions& options, const std::string& name) {
    const OrtApi& api = Ort::GetApi();
    if (name == "cuda") {
        OrtCUDAProviderOptionsV2* raw = nullptr;
        Ort::ThrowOnError(api.CreateCUDAProviderOptions(&raw));
        const auto release = [](OrtCUDAProviderOptionsV2* p) { Ort::GetApi().ReleaseCUDAProviderOptions(p); };
        std::unique_ptr<OrtCUDAProviderOptionsV2, decltype(release)> guard(raw, release);
        options.AppendExecutionProvider_CUDA_V2(*guard);
    } else if (name == "tensorrt") {
        OrtTensorRTProviderOptionsV2* raw = nullptr;
        Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&raw));
        const auto release = [](OrtTensorRTProviderOptionsV2* p) { Ort::GetApi().ReleaseTensorRTProviderOptions(p); };
        std::unique_ptr<OrtTensorRTProviderOptionsV2, decltype(release)> guard(raw, release);
        options.AppendExecutionProvider_TensorRT_V2(*guard);
    } else if (name == "openvino") {
        options.AppendExecutionProvider_OpenVINO_V2({});
    } else if (name == "coreml") {
        options.AppendExecutionProvider("CoreML", {});
    } else if (name == "xnnpack") {
        options.AppendExecutionProvider("XNNPACK", {});
    } else if (name == "nnapi") {
        options.AppendExecutionProvider("NNAPI", {});
    } else {
        log::warn("unknown execution provider '", name, "'");
    }
}

}  // namespace

Ort::Session load_session(const std::string& path, const SessionConfig& config) {
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(config.intra_threads);
    options.SetInterOpNumThreads(config.inter_threads);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    for (const std::string& provider : config.providers) {
        if (provider == "cpu") continue;
        try {
            append_provider(options, provider);
        } catch (const Ort::Exception& e) {
            log::warn("execution provider '", provider, "' unavailable (", e.what(), "); falling back");
        }
    }
    const std::filesystem::path model(path);
    return Ort::Session(env(), model.c_str(), options);
}

}  // namespace ee::onnx
