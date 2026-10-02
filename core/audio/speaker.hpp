#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "core/prosody/prosody_types.hpp"
#include "core/runtime/model_registry.hpp"
#include "core/runtime/stage.hpp"

namespace ee {

/// Speaker encoder slot (blueprint 1.4: ECAPA-TDNN, 192-d voice print).
class ISpeakerEncoder {
public:
    virtual ~ISpeakerEncoder() = default;
    [[nodiscard]] virtual SpeakerEmbedding embed(std::span<const float> audio, int sample_rate) = 0;
};

/// Stand-in until the ECAPA-TDNN export lands (roadmap phase 3): a voice print that carries the
/// speaker's pitch statistics in its first slots, which is what the formant TTS needs to keep
/// the original speaker's pitch. Layout: [0] median F0 / 1000 Hz, [1] F0 spread in
/// semitones / 12, [2] spectral-tilt ratio in dB / 20, rest zero.
class PitchSpeakerEncoder final : public ISpeakerEncoder {
public:
    [[nodiscard]] SpeakerEmbedding embed(std::span<const float> audio, int sample_rate) override;
};

/// Median F0 (Hz) stored by PitchSpeakerEncoder, or 0 if the print carries none.
[[nodiscard]] float voice_print_f0(const SpeakerEmbedding& print) noexcept;

/// `encoder: pitch` (default) or `ecapa` (ECAPA-TDNN through ONNX Runtime, `model`/`model_id`
/// from ml/export/export_ecapa_onnx.py; an L2-normalized 192-d print).
[[nodiscard]] std::unique_ptr<ISpeakerEncoder> make_speaker_encoder(const Params& params, const ModelRegistry* registry);

#if defined(EE_HAVE_ONNXRUNTIME)
[[nodiscard]] std::unique_ptr<ISpeakerEncoder> make_ecapa_encoder(const std::string& path, const Params& params,
                                                                  const ModelRegistry* registry);
#endif

/// Cosine similarity of two voice prints; 0 when either is empty (all zeros).
[[nodiscard]] float voice_similarity(const SpeakerEmbedding& a, const SpeakerEmbedding& b) noexcept;

/// Speaker-aware barge-in (phase 4) on ECAPA voice prints: is speech that starts while the
/// translation plays the interpreted speaker talking on, the translation's own voice coming back
/// through the microphone, or someone else?
class VoiceGate {
public:
    struct Config {
        float threshold = 0.25f;       ///< cosine to the speaker below which a voice is someone else
        float echo_threshold = 0.45f;  ///< cosine to the output voice above which speech is its echo
        int min_enrolled = 3;          ///< utterances in the speaker's voice before judging anyone
        int max_enrolled = 8;          ///< the reference averages up to this many utterances
    };
    enum class Verdict { Unknown, Speaker, Echo, Other };

    explicit VoiceGate(Config config) : cfg_(config) {}

    /// An utterance's print. The first enrols the interpreted speaker; later ones in the same
    /// voice refine the reference, other voices leave it alone.
    void enroll(const SpeakerEmbedding& print);
    /// A print of the translation's own synthesized voice (the latest one counts most).
    void hear_output(const SpeakerEmbedding& print);
    /// Enough of the speaker is enrolled to judge (min_enrolled utterances). A reference from
    /// one or two utterances wrongly rejects the speaker far more often (eval_speaker_verification.py).
    [[nodiscard]] bool has_reference() const noexcept { return enrolled_ >= cfg_.min_enrolled; }
    /// Unknown without a reference or for an empty probe (too little audio to embed).
    [[nodiscard]] Verdict judge(const SpeakerEmbedding& probe, float* to_speaker = nullptr,
                                float* to_output = nullptr) const;

private:
    Config cfg_;
    SpeakerEmbedding reference_{};
    SpeakerEmbedding output_{};
    int enrolled_ = 0;
    bool have_output_ = false;
};

/// Stage 1.4: embeds each utterance's first seconds of speech and emits a VoicePrint frame at
/// the endpoint, which the TTS (4.2) uses to keep the speaker's voice.
///
/// With `barge_in: true` (ECAPA prints only) it also decides barge-in. Speech that starts while
/// the translation plays is embedded after `barge_in_ms` and judged by a VoiceGate. Only another
/// voice cancels the translation: a Control kBargeIn frame to the TTS and playback. The
/// interpreted speaker talking on, or the translation's echo, does not. The translation's voice
/// comes from the TTS's own audio, on a feedback edge.
class SpeakerStage final : public IStage {
public:
    void open(StageContext& ctx) override;
    void process(Frame& frame) override;

private:
    void hear_output(const Frame& f);
    void decide_barge_in(const Frame& f);

    StageContext* ctx_ = nullptr;
    std::unique_ptr<ISpeakerEncoder> encoder_;
    bool pitch_encoder_ = true;  ///< the print itself carries F0 (PitchSpeakerEncoder layout)
    std::vector<float> audio_;
    std::uint64_t utterance_ = 0;
    std::size_t max_samples_ = 0;
    int rate_ = 16000;

    bool barge_in_ = false;
    std::unique_ptr<VoiceGate> gate_;
    std::size_t probe_samples_ = 0;
    bool probing_ = false;         ///< this utterance started over the translation, undecided
    bool not_speaker_ = false;     ///< judged another voice or the echo: not enrolled
    std::vector<float> output_audio_;
    int output_rate_ = 0;
    std::uint64_t output_utterance_ = 0;
    bool output_heard_ = false;    ///< this translation's voice is already embedded
};

}  // namespace ee
