// whisper.cpp adapter (blueprint 2.1). Compiled only with -DEE_WITH_WHISPER=ON.
#include <ggml-backend.h>
#include <whisper.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "core/asr/asr_engine.hpp"
#include "core/runtime/log.hpp"

namespace ee {

namespace {

void forward_whisper_log(ggml_log_level level, const char* text, void* /*user_data*/) {
    std::string_view line(text);
    while (!line.empty() && line.back() == '\n') line.remove_suffix(1);
    if (level == GGML_LOG_LEVEL_ERROR) {
        log::error("whisper: ", line);
    } else if (level == GGML_LOG_LEVEL_WARN) {
        log::warn("whisper: ", line);
    } else if (level == GGML_LOG_LEVEL_INFO && !line.empty()) {
        log::debug("whisper: ", line);
    }
}

/// The first GPU device ggml was built with (CUDA, Metal, Vulkan, ...), if any.
const char* ggml_gpu_name() {
    for (std::size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) return ggml_backend_dev_description(dev);
    }
    return nullptr;
}

// Whisper's encoder always sees a 30 s window (1500 positions, 50 per second). Utterances are a
// few seconds long, so `audio_ctx: auto` encodes a shorter window: the audio plus a margin in
// 128-position steps, never below kMinAutoCtx. Much shorter windows are fragile. On jfk.wav,
// 256 lost a 0.7 s "Ask not" and 384 hallucinated "[MUSIC]" into a fallback loop, while 512
// matched the full window at ~3x less compute. `audio_ctx: 0` keeps the full window.
constexpr int kFullAudioCtx = 1500;
constexpr int kMinAutoCtx = 512;
constexpr int kCtxPerSecond = 50;

int auto_audio_ctx(std::size_t samples) {
    const int needed = static_cast<int>((samples * kCtxPerSecond + WHISPER_SAMPLE_RATE - 1) / WHISPER_SAMPLE_RATE);
    const int padded = (needed + 64 + 127) / 128 * 128;  // >= 1.3 s of margin
    return std::clamp(padded, kMinAutoCtx, kFullAudioCtx);
}

class WhisperEngine final : public IAsrEngine {
public:
    WhisperEngine(const std::string& model_path, const Params& params)
        : threads_(static_cast<int>(params.integer("threads", 4))),
          beam_(static_cast<int>(params.integer("beam", 1))) {
        const std::string ctx = params.str("audio_ctx", "auto");
        if (ctx != "auto") {
            audio_ctx_ = static_cast<int>(params.integer("audio_ctx", 0));
            if (audio_ctx_ < 0 || audio_ctx_ > kFullAudioCtx) {
                throw ConfigError("asr: audio_ctx must be auto or 0.." + std::to_string(kFullAudioCtx));
            }
        }
        whisper_log_set(forward_whisper_log, nullptr);
        whisper_context_params cparams = whisper_context_default_params();
        // A GPU backend is used only when whisper.cpp was built with one (EE_WHISPER_CUDA,
        // Metal, ...); `device: cpu` keeps a GPU build on the CPU.
        const std::string device = params.str("device", "auto");
        cparams.use_gpu = params.flag("gpu", device != "cpu");
        cparams.gpu_device = static_cast<int>(params.integer("gpu_id", 0));
        ctx_ = whisper_init_from_file_with_params(model_path.c_str(), cparams);
        if (ctx_ == nullptr) throw ConfigError("cannot load whisper model '" + model_path + "'");
        if (cparams.use_gpu) {
            if (const char* gpu = ggml_gpu_name()) {
                log::info("asr: whisper.cpp on ", gpu);
            } else if (device == "cuda") {
                log::warn("asr: whisper.cpp was built without a GPU backend (EE_WHISPER_CUDA); using the CPU");
            }
        }
        if (params.flag("warmup", true)) {
            // One throwaway decode at load: the first utterance should not pay for first-touch
            // allocations and thread start-up.
            const std::vector<float> silence(WHISPER_SAMPLE_RATE, 0.0f);
            AsrRequest warm;
            warm.audio = silence;
            warm.final = true;
            warm.language = "en";
            warm.utterance_start_s = -1.0;
            (void)transcribe(warm);
        }
    }
    ~WhisperEngine() override { whisper_free(ctx_); }
    WhisperEngine(const WhisperEngine&) = delete;
    WhisperEngine& operator=(const WhisperEngine&) = delete;

    AsrResult transcribe(const AsrRequest& r) override {
        if (r.sample_rate != WHISPER_SAMPLE_RATE) throw std::runtime_error("whisper.cpp needs 16 kHz audio");
        whisper_full_params wp =
            whisper_full_default_params(beam_ > 1 ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY);
        if (beam_ > 1) wp.beam_search.beam_size = beam_;
        wp.n_threads = threads_;
        wp.print_progress = false;
        wp.print_realtime = false;
        wp.print_special = false;
        wp.print_timestamps = false;
        wp.no_context = true;       // each call re-decodes the whole utterance
        wp.single_segment = false;
        wp.token_timestamps = true;  // word-level timestamps for emphasis alignment (3.1)
        wp.suppress_blank = true;
        wp.suppress_nst = true;  // no "[MUSIC]" / "(laughs)" annotations in the transcript
        // A partial cut mid-word often decodes with low confidence, and temperature fallback would
        // then re-decode it up to five times while the endpoint waits. Only finals get fallback.
        if (!r.final) wp.temperature_inc = 0.0f;
        wp.audio_ctx = audio_ctx_ < 0 ? auto_audio_ctx(r.audio.size()) : audio_ctx_;

        // Automatic language ID costs a second encoder pass, so it runs on the first decode of an
        // utterance only; the partials and the final that follow reuse the detected language.
        const bool detect = r.language.empty() || r.language == "auto";
        if (r.utterance_start_s != utterance_start_s_) {
            utterance_start_s_ = r.utterance_start_s;
            utterance_language_.clear();
        }
        language_ = !detect ? std::string(r.language)
                            : (utterance_language_.empty() ? std::string("auto") : utterance_language_);
        wp.language = language_.c_str();  // "auto" = automatic language ID
        wp.detect_language = false;       // true would detect and stop without transcribing

        if (whisper_full(ctx_, wp, r.audio.data(), static_cast<int>(r.audio.size())) != 0) {
            throw std::runtime_error("whisper_full failed");
        }
        AsrResult out;
        const int lang = whisper_full_lang_id(ctx_);
        out.language = lang >= 0 ? whisper_lang_str(lang) : language_;
        if (detect) utterance_language_ = out.language;
        const whisper_token eot = whisper_token_eot(ctx_);
        for (int s = 0; s < whisper_full_n_segments(ctx_); ++s) {
            for (int t = 0; t < whisper_full_n_tokens(ctx_, s); ++t) {
                const whisper_token_data d = whisper_full_get_token_data(ctx_, s, t);
                if (d.id >= eot) continue;  // timestamps, language and control tokens
                const std::string piece = whisper_full_get_token_text(ctx_, s, t);
                if (piece.empty()) continue;
                const float t0 = static_cast<float>(d.t0) * 0.01f;  // centiseconds
                const float t1 = static_cast<float>(d.t1) * 0.01f;
                if (piece.front() == ' ' || out.words.empty()) {
                    out.words.push_back({piece.substr(piece.front() == ' ' ? 1 : 0), t0, t1, d.p});
                } else {
                    Word& w = out.words.back();
                    w.text += piece;
                    w.t1 = std::max(w.t1, t1);
                    w.probability = std::min(w.probability, d.p);
                }
            }
        }
        std::erase_if(out.words, [](const Word& w) { return w.text.empty(); });
        return out;
    }

private:
    whisper_context* ctx_ = nullptr;
    int threads_;
    int beam_;
    int audio_ctx_ = -1;  ///< -1 = auto
    std::string language_;
    double utterance_start_s_ = -1.0;
    std::string utterance_language_;  ///< language detected for the current utterance
};

}  // namespace

std::unique_ptr<IAsrEngine> make_whisper_engine(const std::string& model_path, const Params& params) {
    return std::make_unique<WhisperEngine>(model_path, params);
}

}  // namespace ee
