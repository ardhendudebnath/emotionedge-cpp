// Acoustic emotion model through ONNX Runtime (blueprint 2.2 "ACOUSTIC: emotion2vec").
// Compiled only with -DEE_WITH_ONNXRUNTIME=ON. Two model contracts:
//
// 1. A classifier directory (ml/export/export_emotion2vec_onnx.py), mapped by ClassEmotionMap:
//      model.onnx   input "waveform" float32 [1, samples] 16 kHz mono in [-1, 1];
//                   output "logits" float32 [1, classes]
//      labels.json  class -> V·A·D map
// 2. A V·A·D regressor file (ml/export/export_acoustic_onnx.py, a trained head):
//      input "waveform"; output "vad" [1, 3] in [-1, 1]; optional "confidence" [1, 3] in [0, 1]
#include <algorithm>
#include <array>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "core/emotion/acoustic.hpp"
#include "core/emotion/class_mapping.hpp"
#include "core/runtime/onnx.hpp"

namespace ee {

namespace {

namespace fs = std::filesystem;

class OnnxAcousticEmotionModel final : public IAcousticEmotionModel {
public:
    OnnxAcousticEmotionModel(const std::string& path, const onnx::SessionConfig& config, const Params& params)
        // Shared: the emotion engine (2.2) and the consistency check (5.2) run the same model.
        : session_(onnx::shared_session(fs::is_directory(path) ? (fs::path(path) / "model.onnx").string() : path, config)),
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
    }

    ModalityEstimate estimate(std::span<const float> audio, std::span<const ProsodyFrame> frames, float hop_s) override {
        (void)frames;
        (void)hop_s;
        if (audio.size() < min_samples_) return {};
        // The end of a long utterance says the most about how it was meant; the cost grows with length.
        if (audio.size() > max_samples_) audio = audio.subspan(audio.size() - max_samples_);
        waveform_.assign(audio.begin(), audio.end());
        const std::array<std::int64_t, 2> dims{1, static_cast<std::int64_t>(waveform_.size())};
        Ort::Value input = Ort::Value::CreateTensor<float>(memory_, waveform_.data(), waveform_.size(), dims.data(), 2);

        std::vector<const char*> outputs{output_.c_str()};
        if (has_confidence_) outputs.push_back(confidence_output_.c_str());
        const char* inputs[] = {input_.c_str()};
        auto result = session_->Run(Ort::RunOptions{nullptr}, inputs, &input, 1, outputs.data(), outputs.size());

        if (map_) {
            const std::size_t n = result[0].GetTensorTypeAndShapeInfo().GetElementCount();
            return map_->from_logits(std::span<const float>(result[0].GetTensorData<float>(), n));
        }
        ModalityEstimate e;
        const float* vad = result[0].GetTensorData<float>();
        e.vad = Vad{vad[0], vad[1], vad[2]}.clamped();
        if (has_confidence_) {
            const float* c = result[1].GetTensorData<float>();
            e.confidence = {std::clamp(c[0], 0.0f, 1.0f), std::clamp(c[1], 0.0f, 1.0f), std::clamp(c[2], 0.0f, 1.0f)};
        } else {
            e.confidence = {0.6f, 0.7f, 0.5f};
        }
        e.valid = true;
        return e;
    }

private:
    std::shared_ptr<Ort::Session> session_;
    Ort::AllocatorWithDefaultOptions allocator_;
    Ort::MemoryInfo memory_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::string input_;
    std::string output_;
    std::string confidence_output_;
    std::size_t min_samples_;
    std::size_t max_samples_;
    bool has_confidence_ = false;
    std::optional<ClassEmotionMap> map_;
    std::vector<float> waveform_;
};

}  // namespace

std::unique_ptr<IAcousticEmotionModel> make_onnx_acoustic_model(const std::string& path, const Params& params,
                                                                const ModelRegistry* registry) {
    return std::make_unique<OnnxAcousticEmotionModel>(path, onnx::session_config(params, registry), params);
}

}  // namespace ee
