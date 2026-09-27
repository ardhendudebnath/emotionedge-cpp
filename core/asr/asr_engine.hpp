#pragma once

#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "core/asr/transcript.hpp"
#include "core/runtime/config.hpp"
#include "core/runtime/model_registry.hpp"

namespace ee {

struct AsrRequest {
    std::span<const float> audio;   ///< mono: the utterance so far, from its first sample
    int sample_rate = 16000;        ///< whisper.cpp requires 16 kHz
    double utterance_start_s = 0.0; ///< input-timeline position of audio[0]
    bool final = false;             ///< endpoint reached: last decode of this utterance
    std::string_view language;      ///< hint; "" or "auto" = detect
};

struct AsrResult {
    std::vector<Word> words;  ///< times relative to audio[0]
    std::string language;     ///< detected (or hinted) language, ISO 639-1
};

/// Speech recognizer slot (blueprint 2.1). Engines re-decode the whole utterance on each call;
/// the streaming stage turns successive hypotheses into stable partials (LocalAgreement-2).
class IAsrEngine {
public:
    virtual ~IAsrEngine() = default;
    [[nodiscard]] virtual AsrResult transcribe(const AsrRequest& request) = 0;
};

/// `engine: scripted` (+ `script`) or `whisper` (+ `model`/`model_id`, `threads`, `beam`).
[[nodiscard]] std::unique_ptr<IAsrEngine> make_asr_engine(const Params& params, const ModelRegistry* registry);

#if defined(EE_HAVE_WHISPER)
[[nodiscard]] std::unique_ptr<IAsrEngine> make_whisper_engine(const std::string& model_path, const Params& params);
#endif

}  // namespace ee
