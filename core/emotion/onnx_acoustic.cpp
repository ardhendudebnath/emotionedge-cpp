// emotion2vec + V·A·D head through ONNX Runtime. Compiled only with -DEE_WITH_ONNXRUNTIME=ON.
//
// Model contract (produced by ml/export/export_acoustic_onnx.py):
//   input  "waveform"   float32 [1, samples]   16 kHz mono in [-1, 1]
//   output "vad"        float32 [1, 3]         valence, arousal, dominance in [-1, 1]
//   output "confidence" float32 [1, 3]         optional per-axis confidence in [0, 1]
#include <algorithm>
#include <array>
#include <string>
#include <vector>

#include "core/emotion/acoustic.hpp"
#include "core/runtime/onnx.hpp"

namespace ee {

namespace {

class OnnxAcousticEmotionModel final : public IAcousticEmotionModel {
public:
    OnnxAcousticEmotionModel(const std::string& path, const onnx::SessionConfig& config, std::string input,
                             std::string output, std::string confidence_output, float min_seconds)
        : session_(onnx::load_session(path, config)), input_(std::move(input)), output_(std::move(output)),
          confidence_output_(std::move(confidence_output)), min_samples_(static_cast<std::size_t>(min_seconds * 16000)) {
        for (std::size_t i = 0; i < session_.GetOutputCount(); ++i) {
            if (session_.GetOutputNameAllocated(i, allocator_).get() == confidence_output_) has_confidence_ = true;
        }
    }

    ModalityEstimate estimate(std::span<const float> audio, std::span<const ProsodyFrame> frames, float hop_s) override {
        (void)frames;
        (void)hop_s;
        ModalityEstimate e;
        if (audio.size() < min_samples_) return e;
        waveform_.assign(audio.begin(), audio.end());
        const std::array<std::int64_t, 2> dims{1, static_cast<std::int64_t>(waveform_.size())};
        Ort::Value input = Ort::Value::CreateTensor<float>(memory_, waveform_.data(), waveform_.size(), dims.data(), 2);

        std::vector<const char*> outputs{output_.c_str()};
        if (has_confidence_) outputs.push_back(confidence_output_.c_str());
        const char* inputs[] = {input_.c_str()};
        auto result = session_.Run(Ort::RunOptions{nullptr}, inputs, &input, 1, outputs.data(), outputs.size());

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
    Ort::Session session_;
    Ort::AllocatorWithDefaultOptions allocator_;
    Ort::MemoryInfo memory_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::string input_;
    std::string output_;
    std::string confidence_output_;
    std::size_t min_samples_;
    bool has_confidence_ = false;
    std::vector<float> waveform_;
};

}  // namespace

std::unique_ptr<IAcousticEmotionModel> make_onnx_acoustic_model(const std::string& path, const Params& params,
                                                                const ModelRegistry* registry) {
    return std::make_unique<OnnxAcousticEmotionModel>(
        path, onnx::session_config(params, registry), params.str("acoustic_input", "waveform"),
        params.str("acoustic_output", "vad"), params.str("acoustic_confidence", "confidence"),
        params.real("acoustic_min_seconds", 0.5f));
}

}  // namespace ee
