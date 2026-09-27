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

/// `encoder: pitch` (default). `ecapa` arrives with the phase-3 model export.
[[nodiscard]] std::unique_ptr<ISpeakerEncoder> make_speaker_encoder(const Params& params, const ModelRegistry* registry);

/// Stage 1.4: embeds each utterance's first seconds of speech and emits a VoicePrint frame at
/// the endpoint, which the TTS (4.2) uses to keep the speaker's voice.
class SpeakerStage final : public IStage {
public:
    void open(StageContext& ctx) override;
    void process(Frame& frame) override;

private:
    StageContext* ctx_ = nullptr;
    std::unique_ptr<ISpeakerEncoder> encoder_;
    std::vector<float> audio_;
    std::uint64_t utterance_ = 0;
    std::size_t max_samples_ = 0;
    int rate_ = 16000;
};

}  // namespace ee
