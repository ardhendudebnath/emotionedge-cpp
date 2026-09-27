// ECAPA-TDNN speaker encoder (blueprint 1.4) through ONNX Runtime. Compiled only with
// -DEE_WITH_ONNXRUNTIME=ON.
//
// Model contract (ml/export/export_ecapa_onnx.py, SpeechBrain spkrec-ecapa-voxceleb):
//   input  "waveform"  float32 [1, samples]  16 kHz mono
//   output "embedding" float32 [192]          L2-normalized
#include <algorithm>
#include <array>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "core/audio/resampler.hpp"
#include "core/audio/speaker.hpp"
#include "core/runtime/onnx.hpp"

namespace ee {

namespace {

namespace fs = std::filesystem;

class EcapaSpeakerEncoder final : public ISpeakerEncoder {
public:
    EcapaSpeakerEncoder(const std::string& path, const onnx::SessionConfig& config)
        : session_(onnx::shared_session(fs::is_directory(path) ? (fs::path(path) / "model.onnx").string() : path,
                                        config)) {}

    SpeakerEmbedding embed(std::span<const float> audio, int sample_rate) override {
        SpeakerEmbedding out{};
        if (audio.empty()) return out;
        wave_ = sample_rate == 16000 ? std::vector<float>(audio.begin(), audio.end())
                                     : Resampler::convert(audio, sample_rate, 16000);
        if (wave_.size() < 8000) return out;  // under 0.5 s says little about a voice
        const std::array<std::int64_t, 2> dims{1, static_cast<std::int64_t>(wave_.size())};
        Ort::Value input = Ort::Value::CreateTensor<float>(memory_, wave_.data(), wave_.size(), dims.data(), 2);
        const char* inputs[] = {"waveform"};
        const char* outputs[] = {"embedding"};
        auto result = session_->Run(Ort::RunOptions{nullptr}, inputs, &input, 1, outputs, 1);
        const float* emb = result[0].GetTensorData<float>();
        const std::size_t n = std::min<std::size_t>(out.size(), result[0].GetTensorTypeAndShapeInfo().GetElementCount());
        std::copy(emb, emb + n, out.begin());
        return out;
    }

private:
    std::shared_ptr<Ort::Session> session_;
    Ort::MemoryInfo memory_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<float> wave_;
};

}  // namespace

std::unique_ptr<ISpeakerEncoder> make_ecapa_encoder(const std::string& path, const Params& params,
                                                    const ModelRegistry* registry) {
    return std::make_unique<EcapaSpeakerEncoder>(path, onnx::session_config(params, registry));
}

}  // namespace ee
