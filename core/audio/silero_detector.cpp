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
    SileroSpeechDetector(const std::string& model_path, const onnx::SessionConfig& config, int sample_rate,
                         double refresh_s, double refresh_warmup_s)
        : session_(onnx::shared_session(model_path, config)), sr_(sample_rate) {
        if (sample_rate != 16000 && sample_rate != 8000) {
            throw ConfigError("Silero VAD runs at 8 or 16 kHz, not " + std::to_string(sample_rate));
        }
        if (refresh_s < 0.0 || refresh_warmup_s < 0.0) {
            throw ConfigError("segmenter: refresh_s and refresh_warmup_ms must be >= 0");
        }
        window_ = sample_rate == 16000 ? 512 : 256;
        context_ = sample_rate == 16000 ? 64 : 32;
        const double window_s = static_cast<double>(window_) / sample_rate;
        refresh_windows_ = static_cast<std::size_t>(refresh_s / window_s + 0.5);
        warmup_windows_ = static_cast<std::size_t>(refresh_warmup_s / window_s + 0.5);
        carried_.clear(window_ + context_);
        if (refresh_windows_ > 0) fresh_.clear(window_ + context_);
    }

    std::size_t window() const noexcept override { return window_; }

    float probability(std::span<const float> window) override {
        const float carried = run(carried_, window);
        if (refresh_windows_ == 0) return carried;
        if (phase_ == SegmenterPhase::Idle && since_refresh_ >= refresh_windows_) {
            fresh_.clear(window_ + context_);
            fresh_age_ = 0;
            since_refresh_ = 0;
        }
        const float fresh = run(fresh_, window);  // always run, so its state follows the audio
        ++fresh_age_;
        ++since_refresh_;
        const bool use_fresh = phase_ != SegmenterPhase::Speech && fresh_age_ > warmup_windows_;
        return use_fresh ? std::max(carried, fresh) : carried;
    }

    void reset() override {
        carried_.clear(window_ + context_);
        if (refresh_windows_ > 0) fresh_.clear(window_ + context_);
        fresh_age_ = 0;
        since_refresh_ = 0;
    }

    void set_phase(SegmenterPhase phase) override { phase_ = phase; }

private:
    static constexpr std::size_t kStateSize = 2 * 1 * 128;

    /// One recurrent state over the stream: the model's state and the context it carries.
    struct Track {
        std::vector<float> input;  ///< the last `context_` samples of the previous window, then this one
        std::vector<float> state;
        std::vector<float> next_state;
        void clear(std::size_t input_size) {
            input.assign(input_size, 0.0f);
            state.assign(kStateSize, 0.0f);
            next_state.assign(kStateSize, 0.0f);
        }
    };

    float run(Track& t, std::span<const float> window) {
        std::copy(window.begin(), window.begin() + static_cast<std::ptrdiff_t>(std::min(window.size(), window_)),
                  t.input.begin() + static_cast<std::ptrdiff_t>(context_));

        const std::array<std::int64_t, 2> input_dims{1, static_cast<std::int64_t>(t.input.size())};
        const std::array<std::int64_t, 3> state_dims{2, 1, 128};
        const std::array<std::int64_t, 1> sr_dims{1};
        const std::array<std::int64_t, 2> prob_dims{1, 1};
        std::array<Ort::Value, 3> inputs{
            Ort::Value::CreateTensor<float>(memory_, t.input.data(), t.input.size(), input_dims.data(), 2),
            Ort::Value::CreateTensor<float>(memory_, t.state.data(), t.state.size(), state_dims.data(), 3),
            Ort::Value::CreateTensor<std::int64_t>(memory_, &sr_, 1, sr_dims.data(), 1)};
        // Outputs are bound to our own buffers, so steady-state inference does not allocate.
        float prob = 0.0f;
        std::array<Ort::Value, 2> outputs{
            Ort::Value::CreateTensor<float>(memory_, &prob, 1, prob_dims.data(), 2),
            Ort::Value::CreateTensor<float>(memory_, t.next_state.data(), t.next_state.size(), state_dims.data(), 3)};

        static constexpr std::array<const char*, 3> kInputNames{"input", "state", "sr"};
        static constexpr std::array<const char*, 2> kOutputNames{"output", "stateN"};
        session_->Run(Ort::RunOptions{nullptr}, kInputNames.data(), inputs.data(), inputs.size(),
                      kOutputNames.data(), outputs.data(), outputs.size());

        std::swap(t.state, t.next_state);
        std::copy(t.input.end() - static_cast<std::ptrdiff_t>(context_), t.input.end(), t.input.begin());
        return prob;
    }

    std::shared_ptr<Ort::Session> session_;  ///< the recurrent states are ours (Track), not the session's
    Ort::MemoryInfo memory_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::int64_t sr_;
    std::size_t window_ = 512;
    std::size_t context_ = 64;
    Track carried_;                   ///< never reset while the stream runs
    Track fresh_;                     ///< reset every refresh_windows_ between utterances
    std::size_t refresh_windows_ = 0; ///< 0: no fresh state
    std::size_t warmup_windows_ = 0;
    std::size_t fresh_age_ = 0;       ///< windows since fresh_ was reset
    std::size_t since_refresh_ = 0;
    SegmenterPhase phase_ = SegmenterPhase::Idle;
};

}  // namespace

std::unique_ptr<ISpeechDetector> make_silero_detector(const std::string& model_path, const Params& params,
                                                      const ModelRegistry* registry, int sample_rate) {
    return std::make_unique<SileroSpeechDetector>(model_path, onnx::session_config(params, registry), sample_rate,
                                                  params.number("refresh_s", 4.0),
                                                  params.number("refresh_warmup_ms", 500.0) / 1000.0);
}

}  // namespace ee
