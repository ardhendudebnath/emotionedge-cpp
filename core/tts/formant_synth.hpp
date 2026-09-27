#pragma once

#include <string_view>

#include "core/tts/tts_engine.hpp"

namespace ee {

/// Voice settings of the formant synthesizer. Defaults sit near the population priors of the
/// prosody emotion model, so a neutral render reads as neutral.
struct FormantVoice {
    int sample_rate = 24000;
    float base_f0_hz = 180.0f;     ///< overridden by the speaker voice print when present
    float level_dbfs = -20.0f;     ///< RMS level of voiced speech
    float rolloff = 1.5f;          ///< harmonic k has amplitude k^-rolloff: the voice's spectral tilt
    float syllable_s = 0.22f;      ///< syllable duration at the neutral rate
    float syllable_dip = 0.25f;    ///< amplitude between syllables of one word (-12 dB)
    float word_gap_s = 0.035f;     ///< silence between words (scaled by hesitation)
    float clause_gap_s = 0.18f;    ///< pause after clause punctuation (scaled by hesitation)
    float range_st = 2.0f;         ///< intonation spread around the declination line
    float declination_st = 2.0f;   ///< pitch drift over a clause
    float final_fall_st = 2.0f;    ///< declarative fall on the last syllable
    float emphasis_db = 4.0f;      ///< extra loudness of emphasized words
};

/// Stand-in expressive TTS: an additive harmonic "vowel" voice that renders every prosody
/// target (pitch mean and range, rate, energy, emphasis accent and pre-pause, final contour,
/// hesitation pauses, voice tension as spectral tilt) and reports word timings. It sounds like
/// a buzzy hum rather than speech, but the emotion engine can measure what it was asked to
/// express, which makes the 5.2 -> 4.1 closed loop testable without neural models.
class FormantSynth final : public ITtsEngine {
public:
    explicit FormantSynth(FormantVoice voice = {});
    [[nodiscard]] int sample_rate() const noexcept override { return voice_.sample_rate; }
    void synthesize(const SynthesisRequest& request, SynthesisResult& out) override;
    [[nodiscard]] const FormantVoice& voice() const noexcept { return voice_; }

private:
    FormantVoice voice_;
};

/// Rough syllable count: vowel groups for Latin script, base characters for Devanagari,
/// half the characters elsewhere. At least 1.
[[nodiscard]] int count_syllables(std::string_view word);

}  // namespace ee
