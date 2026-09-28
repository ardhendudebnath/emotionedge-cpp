#include "core/runtime/onnx.hpp"

#include <algorithm>
#include <array>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>

#include "core/runtime/log.hpp"

namespace ee::onnx {

SessionConfig session_config(const Params& params, const ModelRegistry* registry, std::string_view prefix) {
    // `<prefix>key` when the stage sets it, else `key` (list values are stored as `key.0`, ...).
    const auto key = [&](std::string_view name) {
        std::string prefixed = std::string(prefix) + std::string(name);
        const bool set = !prefix.empty() && (params.has(prefixed) || params.has(prefixed + ".0"));
        return set ? prefixed : std::string(name);
    };
    SessionConfig cfg;
    std::string device = params.str(key("device"), "cpu");
    // auto: CUDA when this ORT build has it. A build that has it but cannot start it (no GPU,
    // CUDA libraries not found) still warns and falls back in load_session.
    if (device == "auto") device = provider_available("cuda") ? "cuda" : "cpu";
    const DeviceProfile* profile = registry != nullptr ? registry->device_profile(device) : nullptr;
    if (profile != nullptr) {
        if (!profile->execution_providers.empty()) cfg.providers = profile->execution_providers;
        if (profile->threads > 0) cfg.intra_threads = profile->threads;
    } else if (device != "cpu") {
        cfg.providers = {device, "cpu"};  // no manifest profile: the device names the provider
    }
    if (const auto providers = params.list(key("providers")); !providers.empty()) cfg.providers = providers;
    cfg.intra_threads = static_cast<int>(params.integer(key("threads"), cfg.intra_threads));
    cfg.spin = params.flag(key("spin"), cfg.spin);
    cfg.gpu_id = static_cast<int>(params.integer(key("gpu_id"), cfg.gpu_id));
    cfg.profile = params.str(key("ort_profile"));
    return cfg;
}

Ort::Env& env() {
    // Never destroyed: releasing the environment during static destruction unloads the CUDA
    // provider library while CUDA's own teardown is under way, which corrupted the heap at exit
    // in about 1 run in 10 on CUDA. The process is exiting anyway.
    // Errors only: ORT's warnings here are by-design notices that repeat on every GPU run
    // (shape ops kept on the CPU, "ScatterND ... if indices are not duplicated").
    static Ort::Env* instance = new Ort::Env(ORT_LOGGING_LEVEL_ERROR, "emotionedge");
    return *instance;
}

const std::vector<std::string>& available_providers() {
    static const std::vector<std::string> providers = Ort::GetAvailableProviders();
    return providers;
}

namespace {

// SessionConfig's short provider names -> ORT's.
std::string ort_provider_name(std::string_view name) {
    if (name == "cpu") return "CPUExecutionProvider";
    if (name == "cuda") return "CUDAExecutionProvider";
    if (name == "tensorrt") return "TensorrtExecutionProvider";
    if (name == "openvino") return "OpenVINOExecutionProvider";
    if (name == "coreml") return "CoreMLExecutionProvider";
    if (name == "xnnpack") return "XnnpackExecutionProvider";
    if (name == "nnapi") return "NnapiExecutionProvider";
    return std::string(name);
}

}  // namespace

bool provider_available(std::string_view name) {
    const std::string wanted = ort_provider_name(name);
    const auto& providers = available_providers();
    return std::find(providers.begin(), providers.end(), wanted) != providers.end();
}

namespace {

void append_provider(Ort::SessionOptions& options, const std::string& name, const SessionConfig& config) {
    const OrtApi& api = Ort::GetApi();
    if (name == "cuda") {
        OrtCUDAProviderOptionsV2* raw = nullptr;
        Ort::ThrowOnError(api.CreateCUDAProviderOptions(&raw));
        const auto release = [](OrtCUDAProviderOptionsV2* p) { Ort::GetApi().ReleaseCUDAProviderOptions(p); };
        std::unique_ptr<OrtCUDAProviderOptionsV2, decltype(release)> guard(raw, release);
        const std::string gpu = std::to_string(config.gpu_id);
        const std::array<const char*, 1> keys{"device_id"};
        const std::array<const char*, 1> values{gpu.c_str()};
        Ort::ThrowOnError(api.UpdateCUDAProviderOptions(guard.get(), keys.data(), values.data(), keys.size()));
        options.AppendExecutionProvider_CUDA_V2(*guard);
    } else if (name == "tensorrt") {
        OrtTensorRTProviderOptionsV2* raw = nullptr;
        Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&raw));
        const auto release = [](OrtTensorRTProviderOptionsV2* p) { Ort::GetApi().ReleaseTensorRTProviderOptions(p); };
        std::unique_ptr<OrtTensorRTProviderOptionsV2, decltype(release)> guard(raw, release);
        const std::string gpu = std::to_string(config.gpu_id);
        const std::array<const char*, 1> keys{"device_id"};
        const std::array<const char*, 1> values{gpu.c_str()};
        Ort::ThrowOnError(api.UpdateTensorRTProviderOptions(guard.get(), keys.data(), values.data(), keys.size()));
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
    // The environment must exist before a provider is appended: the CUDA provider logs through
    // ORT's default logger, which the environment registers. Otherwise the first session of a
    // process cannot start CUDA ("Attempt to use DefaultLogger but none has been registered").
    Ort::Env& environment = env();
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(config.intra_threads);
    options.SetInterOpNumThreads(config.inter_threads);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    options.AddConfigEntry("session.intra_op.allow_spinning", config.spin ? "1" : "0");
    options.AddConfigEntry("session.inter_op.allow_spinning", config.spin ? "1" : "0");
    if (!config.profile.empty()) options.EnableProfiling(std::filesystem::path(config.profile).c_str());
    // Every available provider is appended in priority order: ORT gives each node to the first
    // that supports it (TensorRT, then CUDA for what TensorRT lacks, then the CPU).
    const std::filesystem::path model(path);
    // "kokoro-82m-hi/model.onnx": exported models share the file name.
    const std::string name = (model.parent_path().filename() / model.filename()).generic_string();
    std::string placed;
    for (const std::string& provider : config.providers) {
        if (provider == "cpu") continue;  // ORT's implicit final fallback
        if (!provider_available(provider)) {
            log::warn(name, ": this ONNX Runtime build has no '", provider, "' execution provider; falling back");
            continue;
        }
        try {
            append_provider(options, provider, config);
            placed += (placed.empty() ? "" : ", ") + provider;
        } catch (const Ort::Exception& e) {
            // Built in, but it failed to start: e.g. the CUDA/cuDNN libraries are not on the
            // library path, or there is no GPU.
            log::warn(name, ": execution provider '", provider, "' unavailable (", e.what(), "); falling back");
        }
    }
    if (!placed.empty()) log::info(name, ": ", placed, " (gpu ", config.gpu_id, ")");
    return Ort::Session(environment, model.c_str(), options);
}

std::shared_ptr<Ort::Session> shared_session(const std::string& path, const SessionConfig& config) {
    static std::mutex mutex;
    static std::map<std::string, std::weak_ptr<Ort::Session>> sessions;
    std::string key = std::filesystem::weakly_canonical(path).string() + '|' + std::to_string(config.intra_threads) +
                      '|' + std::to_string(config.inter_threads) + '|' + (config.spin ? "spin" : "idle") + '|' +
                      std::to_string(config.gpu_id) + '|' + config.profile;
    for (const std::string& provider : config.providers) key += '|' + provider;
    std::lock_guard lock(mutex);
    if (auto existing = sessions[key].lock()) return existing;
    auto session = std::make_shared<Ort::Session>(load_session(path, config));
    sessions[key] = session;
    return session;
}

}  // namespace ee::onnx
