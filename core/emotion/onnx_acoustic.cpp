// Acoustic emotion model through ONNX Runtime (blueprint 2.2 "ACOUSTIC: emotion2vec").
// Compiled only with -DEE_WITH_ONNXRUNTIME=ON. Two model contracts:
//
// 1. A classifier directory (ml/export/export_emotion2vec_onnx.py), mapped by ClassEmotionMap:
//      model.onnx   input "waveform" float32 [batch, samples] 16 kHz mono in [-1, 1];
//                   output "logits" float32 [batch, classes] (an export fixed to batch 1 works,
//                   unbatched)
//      labels.json  class -> V·A·D map
// 2. A V·A·D regressor file (ml/export/export_acoustic_onnx.py, a trained head):
//      input "waveform"; output "vad" [1, 3] in [-1, 1]; optional "confidence" [1, 3] in [0, 1]
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "core/emotion/acoustic.hpp"
#include "core/emotion/class_mapping.hpp"
#include "core/runtime/batcher.hpp"
#include "core/runtime/log.hpp"
#include "core/runtime/onnx.hpp"
#include "core/runtime/shared_cache.hpp"

namespace ee {

namespace {

namespace fs = std::filesystem;

/// One row's values of each output.
using Outputs = std::vector<std::vector<float>>;

/// Runs waveforms of one length as one batch of rows; returns each row's outputs.
std::vector<Outputs> run_rows(Ort::Session& session, const char* input, const std::vector<const char*>& outputs,
                              const std::vector<std::vector<float>>& rows) {
    const std::size_t samples = rows.front().size();
    std::vector<float> data;
    data.reserve(rows.size() * samples);
    for (const auto& row : rows) data.insert(data.end(), row.begin(), row.end());
    const std::array<std::int64_t, 2> dims{static_cast<std::int64_t>(rows.size()), static_cast<std::int64_t>(samples)};
    const Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value tensor = Ort::Value::CreateTensor<float>(memory, data.data(), data.size(), dims.data(), 2);
    const char* inputs[] = {input};
    auto result = session.Run(Ort::RunOptions{nullptr}, inputs, &tensor, 1, outputs.data(), outputs.size());
    std::vector<Outputs> out(rows.size(), Outputs(outputs.size()));
    for (std::size_t o = 0; o < outputs.size(); ++o) {
        const std::size_t per_row = result[o].GetTensorTypeAndShapeInfo().GetElementCount() / rows.size();
        const float* values = result[o].GetTensorData<float>();
        for (std::size_t i = 0; i < rows.size(); ++i) out[i][o].assign(values + i * per_row, values + (i + 1) * per_row);
    }
    return out;
}

/// The model as every session of the process shares it on a GPU (its emotion engine and its
/// consistency check): waveforms of one length that arrive together, such as every session's
/// 1 s partial estimates, run as one batch. Each row's normalization and pooling are its own, so
/// a row gets what it would alone (ml/export/export_emotion2vec_onnx.py checks it).
struct SharedAcoustic {
    std::shared_ptr<Ort::Session> session;
    std::string input;
    std::vector<std::string> outputs;
    std::unique_ptr<Batcher<std::vector<float>, Outputs>> batcher;  ///< last: it runs on the above
};

std::string model_file(const std::string& path) {
    return fs::is_directory(path) ? (fs::path(path) / "model.onnx").string() : path;
}

class OnnxAcousticEmotionModel final : public IAcousticEmotionModel {
public:
    OnnxAcousticEmotionModel(const std::string& path, const onnx::SessionConfig& config, const Params& params)
        // Shared: the emotion engine (2.2) and the consistency check (5.2) run the same model.
        : session_(onnx::shared_session(model_file(path), config)),
          input_(params.str("acoustic_input", "waveform")),
          min_samples_(static_cast<std::size_t>(params.real("acoustic_min_seconds", 0.5f) * 16000)),
          max_samples_(static_cast<std::size_t>(params.real("acoustic_max_seconds", 8.0f) * 16000)) {
        if (fs::is_directory(path)) {
            map_ = ClassEmotionMap::from_file(fs::path(path) / "labels.json");
            output_ = params.str("acoustic_output", "logits");
        } else {
            output_ = params.str("acoustic_output", "vad");
            confidence_output_ = params.str("acoustic_confidence", "confidence");
            for (std::size_t i = 0; i < session_->GetOutputCount(); ++i) {
                if (session_->GetOutputNameAllocated(i, allocator_).get() == confidence_output_) has_confidence_ = true;
            }
        }
        outputs_.push_back(output_.c_str());
        if (has_confidence_) outputs_.push_back(confidence_output_.c_str());
        share_batches(model_file(path), config, params);
        // The first run pays for allocations (and, on a GPU, kernel loading): not the first utterance.
        if (params.flag("warmup", true)) {
            std::vector<float> tone(std::max<std::size_t>(min_samples_, 16000));
            for (std::size_t i = 0; i < tone.size(); ++i) tone[i] = 0.1f * std::sin(0.0864f * static_cast<float>(i));
            (void)estimate(tone, {}, 0.01f);
        }
    }

    ModalityEstimate estimate(std::span<const float> audio, std::span<const ProsodyFrame> frames, float hop_s) override {
        (void)frames;
        (void)hop_s;
        if (audio.size() < min_samples_) return {};
        // The end of a long utterance says the most about how it was meant; the cost grows with length.
        if (audio.size() > max_samples_) audio = audio.subspan(audio.size() - max_samples_);
        std::vector<float> waveform(audio.begin(), audio.end());
        Outputs out;
        if (shared_) {
            out = shared_->batcher->submit(std::move(waveform));
        } else {
            std::vector<std::vector<float>> row;
            row.push_back(std::move(waveform));
            out = std::move(run_rows(*session_, input_.c_str(), outputs_, row).front());
        }

        if (map_) return map_->from_logits(out[0]);
        ModalityEstimate e;
        const std::vector<float>& vad = out[0];
        e.vad = Vad{vad[0], vad[1], vad[2]}.clamped();
        if (has_confidence_) {
            const std::vector<float>& c = out[1];
            e.confidence = {std::clamp(c[0], 0.0f, 1.0f), std::clamp(c[1], 0.0f, 1.0f), std::clamp(c[2], 0.0f, 1.0f)};
        } else {
            e.confidence = {0.6f, 0.7f, 0.5f};
        }
        e.valid = true;
        return e;
    }

private:
    /// `acoustic_batch`: auto (on a GPU, shared by several sessions), on or off. Needs an export
    /// with a dynamic batch axis; one fixed to 1 runs unbatched (or, with `on`, is refused).
    void share_batches(const std::string& file, const onnx::SessionConfig& config, const Params& params) {
        const std::string mode = params.str("acoustic_batch", "auto");
        if (mode != "auto" && mode != "on" && mode != "off") throw ConfigError("acoustic_batch must be auto, on or off");
        const std::int64_t sessions = std::max<std::int64_t>(1, params.integer("sessions", 1));
        const bool gpu = !config.providers.empty() && config.providers.front() != "cpu" &&
                         onnx::provider_available(config.providers.front());
        if (mode == "off" || (mode == "auto" && (!gpu || sessions < 2))) return;
        const std::vector<std::int64_t> shape = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        if (shape.empty() || shape.front() >= 0) {
            if (mode == "on") throw ConfigError("acoustic_batch: on needs an export with a dynamic batch axis: " + file);
            static std::atomic<bool> told{false};
            if (!told.exchange(true)) {
                log::info("emotion: ", file, " takes one waveform at a time; re-export it ",
                          "(ml/export/export_emotion2vec_onnx.py) to batch them across sessions");
            }
            return;
        }
        static SharedCache<SharedAcoustic> cache;
        std::string key = file + '|' + output_ + '|' + confidence_output_;
        for (const std::string& p : config.providers) key += '|' + p;
        shared_ = cache.get(key, [&] {
            auto s = std::make_shared<SharedAcoustic>();
            s->session = session_;
            s->input = input_;
            for (const char* o : outputs_) s->outputs.emplace_back(o);
            SharedAcoustic* raw = s.get();
            s->batcher = std::make_unique<Batcher<std::vector<float>, Outputs>>(
                [raw](std::vector<std::vector<float>>& rows) {
                    std::vector<const char*> names;
                    for (const std::string& o : raw->outputs) names.push_back(o.c_str());
                    return run_rows(*raw->session, raw->input.c_str(), names, rows);
                },
                static_cast<std::size_t>(2 * sessions),  // each session's emotion engine and consistency check
                [](const std::vector<float>& row) { return static_cast<int>(row.size()); });
            return s;
        });
    }

    std::shared_ptr<Ort::Session> session_;
    Ort::AllocatorWithDefaultOptions allocator_;
    std::string input_;
    std::string output_;
    std::string confidence_output_;
    std::vector<const char*> outputs_;  ///< output_ (and confidence_output_)
    std::size_t min_samples_;
    std::size_t max_samples_;
    bool has_confidence_ = false;
    std::optional<ClassEmotionMap> map_;
    std::shared_ptr<SharedAcoustic> shared_;  ///< batches across sessions, when they do
};

}  // namespace

std::unique_ptr<IAcousticEmotionModel> make_onnx_acoustic_model(const std::string& path, const Params& params,
                                                                const ModelRegistry* registry) {
    return std::make_unique<OnnxAcousticEmotionModel>(path, onnx::session_config(params, registry, "acoustic_"), params);
}

}  // namespace ee
