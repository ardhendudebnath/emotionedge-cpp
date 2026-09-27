#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "core/asr/transcript.hpp"
#include "core/prosody/prosody_types.hpp"
#include "core/runtime/model_registry.hpp"
#include "core/runtime/params.hpp"

namespace ee {

/// One clause to speak.
struct SynthesisRequest {
    std::string_view text;
    std::string_view language;
    ProsodyTargets prosody;
    std::vector<std::uint16_t> emphasis;       ///< clause-local word indices
    const StyleVector* style = nullptr;        ///< StyleTTS2 conditioning (4.1)
    const SpeakerEmbedding* voice = nullptr;   ///< speaker voice print (1.4)
    bool utterance_final = true;               ///< last clause: apply the final contour
};

struct SynthesisResult {
    std::vector<float> audio;
    int sample_rate = 0;
    std::vector<Word> words;  ///< word timings in `audio`, when the engine knows them
};

/// Expressive TTS slot (blueprint 4.2: StyleTTS2 + HiFi-GAN; fast fallback Piper / VITS).
class ITtsEngine {
public:
    virtual ~ITtsEngine() = default;
    [[nodiscard]] virtual int sample_rate() const noexcept = 0;
    /// Synthesizes into `out`, reusing its buffers.
    virtual void synthesize(const SynthesisRequest& request, SynthesisResult& out) = 0;
};

/// `engine: formant` (default) or `piper` (+ `model`/`model_id`, the voice's .onnx with its
/// .onnx.json next to it).
[[nodiscard]] std::unique_ptr<ITtsEngine> make_tts_engine(const Params& params, const ModelRegistry* registry);

#if defined(EE_HAVE_PIPER)
[[nodiscard]] std::unique_ptr<ITtsEngine> make_piper_engine(const std::string& model_path, const Params& params,
                                                            const ModelRegistry* registry);
#endif

}  // namespace ee
