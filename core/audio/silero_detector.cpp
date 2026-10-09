// Silero VAD v5/v6 through ONNX Runtime. Compiled only with -DEE_WITH_ONNXRUNTIME=ON.
#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <vector>

#include "core/audio/speech_detector.hpp"
#include "core/runtime/onnx.hpp"

namespace ee {

namespace {

class SileroSpeechDetector final : public ISpeechDetector {
public:
    SileroSpeechDetector(const std::string& model_path, const onnx::SessionConfig& config, int sample_rate)
        : session_(onnx::shared_session(model_path, config)), sr_(sample_rate) {
        if (sample_rate != 16000 && sample_rate != 8000) {
            throw ConfigError("Silero VAD runs at 8 or 16 kHz, not " + std::to_string(sample_rate));
        }
        window_ = sample_rate == 16000 ? 512 : 256;
        context_ = sample_rate == 16000 ? 64 : 32;
        input_.assign(window_ + context_, 0.0f);
        state_.assign(kStateSize, 0.0f);
        next_state_.assign(kStateSize, 0.0f);
    }

    std::size_t window() const noexcept override { return window_; }

    float probability(std::span<const float> window) override {
        // Input = last `context_` samples of the previous window followed by this window.
        std::copy(window.begin(), window.begin() + static_cast<std::ptrdiff_t>(std::min(window.size(), window_)),
                  input_.begin() + static_cast<std::ptrdiff_t>(context_));

        const std::array<std::int64_t, 2> input_dims{1, static_cast<std::int64_t>(input_.size())};
        const std::array<std::int64_t, 3> state_dims{2, 1, 128};
        const std::array<std::int64_t, 1> sr_dims{1};
        const std::array<std::int64_t, 2> prob_dims{1, 1};
        std::array<Ort::Value, 3> inputs{
            Ort::Value::CreateTensor<float>(memory_, input_.data(), input_.size(), input_dims.data(), 2),
            Ort::Value::CreateTensor<float>(memory_, state_.data(), state_.size(), state_dims.data(), 3),
            Ort::Value::CreateTensor<std::int64_t>(memory_, &sr_, 1, sr_dims.data(), 1)};
        // Outputs are bound to our own buffers, so steady-state inference does not allocate.
        std::array<Ort::Value, 2> outputs{
            Ort::Value::CreateTensor<float>(memory_, &prob_, 1, prob_dims.data(), 2),
            Ort::Value::CreateTensor<float>(memory_, next_state_.data(), next_state_.size(), state_dims.data(), 3)};

        static constexpr std::array<const char*, 3> kInputNames{"input", "state", "sr"};
        static constexpr std::array<const char*, 2> kOutputNames{"output", "stateN"};
        session_->Run(Ort::RunOptions{nullptr}, kInputNames.data(), inputs.data(), inputs.size(),
                      kOutputNames.data(), outputs.data(), outputs.size());

        std::swap(state_, next_state_);
        std::copy(input_.end() - static_cast<std::ptrdiff_t>(context_), input_.end(), input_.begin());
        return prob_;
    }

    void reset() override {
        std::fill(input_.begin(), input_.end(), 0.0f);
        std::fill(state_.begin(), state_.end(), 0.0f);
    }

private:
    static constexpr std::size_t kStateSize = 2 * 1 * 128;
    std::shared_ptr<Ort::Session> session_;  ///< the recurrent state is ours (state_), not the session's
    Ort::MemoryInfo memory_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::int64_t sr_;
    std::size_t window_ = 512;
    std::size_t context_ = 64;
    std::vector<float> input_;
    std::vector<float> state_;
    std::vector<float> next_state_;
    float prob_ = 0.0f;
};

}  // namespace

std::unique_ptr<ISpeechDetector> make_silero_detector(const std::string& model_path, const Params& params,
                                                      const ModelRegistry* registry, int sample_rate) {
    return std::make_unique<SileroSpeechDetector>(model_path, onnx::session_config(params, registry), sample_rate);
}

}  // namespace ee
