#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include "core/runtime/params.hpp"

namespace ee {

/// One model file listed in models/manifest.json (written by ml/export/write_manifest.py).
struct ModelEntry {
    std::string id;         ///< e.g. "asr.whisper.base.q5_1"
    std::string task;       ///< asr | vad | emotion.acoustic | emotion.lexical | speaker | mt | tts
    std::string format;     ///< onnx | ggml | ct2 | piper
    std::string path;       ///< as written in the manifest, relative to the manifest directory
    std::string check_file; ///< for directory models (CTranslate2): the file inside that is verified
    std::string sha256;     ///< lowercase hex of the verified file; empty = not pinned yet
    std::uint64_t bytes = 0;
    std::vector<std::string> languages;  ///< ISO 639-1 codes, "*" for any
    std::vector<std::string> devices;    ///< cpu | cuda | npu ...
    std::string precision;
    std::string license;
    std::string source;
    std::filesystem::path resolved;  ///< absolute path on disk
};

/// Execution preferences per device class (blueprint "Model Registry: device profiles").
struct DeviceProfile {
    std::string name;
    std::vector<std::string> execution_providers;  ///< ORT EPs in priority order
    int threads = 0;                               ///< 0 = runtime default
    std::string precision;                         ///< int8 | fp16 | fp32
};

/// The ASR / MT / TTS models that serve one language pair.
struct LanguagePack {
    std::string source;
    std::string target;
    std::map<std::string, std::string, std::less<>> models;  ///< task -> model id
};

enum class VerifyStatus { Ok, Missing, SizeMismatch, HashMismatch, Unpinned };
[[nodiscard]] std::string_view to_string(VerifyStatus status) noexcept;

struct VerifyResult {
    const ModelEntry* entry = nullptr;
    VerifyStatus status = VerifyStatus::Missing;
    std::string actual_sha256;
};

class ModelRegistry {
public:
    /// Throws ConfigError on unreadable or malformed manifests.
    [[nodiscard]] static ModelRegistry load(const std::filesystem::path& manifest);
    [[nodiscard]] static ModelRegistry parse(std::string_view json, const std::filesystem::path& base_dir);

    [[nodiscard]] const std::vector<ModelEntry>& models() const noexcept { return models_; }
    [[nodiscard]] const ModelEntry* find(std::string_view id) const;
    /// The model for `task` from the source->target language pack, else the first model of
    /// that task whose languages include the relevant language (or "*").
    [[nodiscard]] const ModelEntry* resolve(std::string_view task, std::string_view source,
                                            std::string_view target) const;
    [[nodiscard]] const LanguagePack* language_pack(std::string_view source, std::string_view target) const;
    [[nodiscard]] const DeviceProfile* device_profile(std::string_view name) const;

    /// Checks existence, size and SHA-256. A `<file>.sha256` stamp recording hash, size and
    /// mtime lets later runs skip rehashing unchanged multi-GB files.
    [[nodiscard]] VerifyResult verify(const ModelEntry& entry, bool use_stamp = true) const;
    [[nodiscard]] std::vector<VerifyResult> verify_all(bool use_stamp = true) const;

private:
    std::vector<ModelEntry> models_;
    std::vector<DeviceProfile> devices_;
    std::vector<LanguagePack> packs_;
};

/// Resolves a stage's model file: `params[key + "_id"]` names a manifest entry (which must
/// verify), otherwise `params[key]` is taken as a path. Returns "" when neither is set.
[[nodiscard]] std::string resolve_model_path(const Params& params, const ModelRegistry* registry,
                                             std::string_view key = "model");

/// Holds the engine for the active language pair. A new engine can be published from any
/// thread; stages acquire() once per utterance, so a swap lands on an utterance boundary
/// (blueprint "hot-swap per language pair"). The lock guards a pointer copy, never inference.
template <typename T>
class HotSwap {
public:
    void publish(std::shared_ptr<T> next) {
        std::lock_guard lock(mutex_);
        current_ = std::move(next);
    }
    [[nodiscard]] std::shared_ptr<T> acquire() const {
        std::lock_guard lock(mutex_);
        return current_;
    }

private:
    mutable std::mutex mutex_;
    std::shared_ptr<T> current_;
};

}  // namespace ee
