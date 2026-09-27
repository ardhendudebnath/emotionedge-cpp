// whisper.cpp adapter (blueprint 2.1). Compiled only with -DEE_WITH_WHISPER=ON.
#include <whisper.h>

#include <algorithm>
#include <stdexcept>
#include <string>

#include "core/asr/asr_engine.hpp"
#include "core/runtime/log.hpp"

namespace ee {

namespace {

void forward_whisper_log(ggml_log_level level, const char* text, void* /*user_data*/) {
    if (level == GGML_LOG_LEVEL_ERROR) {
        log::error("whisper: ", text);
    } else if (level == GGML_LOG_LEVEL_WARN) {
        log::warn("whisper: ", text);
    }
}

class WhisperEngine final : public IAsrEngine {
public:
    WhisperEngine(const std::string& model_path, const Params& params)
        : threads_(static_cast<int>(params.integer("threads", 4))),
          beam_(static_cast<int>(params.integer("beam", 1))) {
        whisper_log_set(forward_whisper_log, nullptr);
        whisper_context_params cparams = whisper_context_default_params();
        cparams.use_gpu = params.flag("gpu", true);
        ctx_ = whisper_init_from_file_with_params(model_path.c_str(), cparams);
        if (ctx_ == nullptr) throw ConfigError("cannot load whisper model '" + model_path + "'");
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
        language_ = r.language.empty() ? std::string("auto") : std::string(r.language);
        wp.language = language_.c_str();  // "auto" = automatic language ID
        wp.detect_language = false;       // true would detect and stop without transcribing

        if (whisper_full(ctx_, wp, r.audio.data(), static_cast<int>(r.audio.size())) != 0) {
            throw std::runtime_error("whisper_full failed");
        }
        AsrResult out;
        const int lang = whisper_full_lang_id(ctx_);
        out.language = lang >= 0 ? whisper_lang_str(lang) : language_;
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
    std::string language_;
};

}  // namespace

std::unique_ptr<IAsrEngine> make_whisper_engine(const std::string& model_path, const Params& params) {
    return std::make_unique<WhisperEngine>(model_path, params);
}

}  // namespace ee
