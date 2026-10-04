#pragma once

#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include "core/emotion/emotion_types.hpp"
#include "core/prosody/expressivity.hpp"
#include "core/prosody/prosody_types.hpp"
#include "core/runtime/params.hpp"

namespace ee {

/// A learned prosody plan (ml/train/learn_controller.py): per emotion, the speaking rate,
/// pitch shift, pitch range and final contour that make the consistency judge (5.2) hear it in
/// one TTS voice. The values are given at full strength. They are searched and checked at the
/// partial strengths the controller mostly applies them at. Only emotions an independent judge
/// also confirmed are listed.
struct LearnedProsody {
    struct Controls {
        float speed = 1.0f;     ///< rate multiplier
        float pitch_st = 0.0f;  ///< pitch-mean shift, semitones
        float range = 1.0f;     ///< pitch-range multiplier
        float fall = 0.0f;      ///< final contour, as ProsodyTargets::final_fall
    };
    std::map<EmotionLabel, Controls> controls;

    /// Reads `controls` restricted to `adopted` from the learner's JSON and, when `only` is not
    /// empty, to those emotions as well. Throws ConfigError.
    [[nodiscard]] static LearnedProsody load(const std::filesystem::path& path, const std::vector<std::string>& only = {});
};

struct ControllerConfig {
    float confidence_floor = 0.30f;  ///< at or below: every target shrinks to neutral
    float confidence_full = 0.75f;   ///< at or above: full-strength targets
    float ecs_threshold = 0.75f;     ///< closed loop engages below this ECS
    float loop_gain = 0.6f;          ///< fraction of -ΔVAD applied per feedback
    float loop_decay = 0.5f;         ///< correction kept at each new utterance
    float max_correction = 0.35f;    ///< per-axis bound on the accumulated correction

    static ControllerConfig from(const Params& params);
};

/// Blueprint 4.1 "Emotion Controller" with the page-3 "Controller rules v1": transparent rules
/// from V·A·D to prosody targets relative to the target language's neutral baseline, to be
/// replaced by a small learned regressor once ECS data accumulates.
///
///   Arousal ↑     wider pitch range, faster rate, more energy (and a higher pitch mean)
///   Valence ↓     lower pitch mean (sad) or tenser voice quality (anger)
///   Dominance ↑   falling final contours, fewer hesitation pauses
///   Emphasis span local pitch accent + 60–100 ms pre-pause
///   Low confidence shrink every target toward neutral
///   Target language scale by its expressivity profile (3.3)
///
/// Closed loop: if the emotion consistency score (5.2) drops below the threshold, the
/// controller nudges its targets along -ΔVAD for the next chunk.
class EmotionController {
public:
    explicit EmotionController(ControllerConfig config = {});

    /// Use a learned plan for the emotions it lists. For those, rate, pitch mean, range and the
    /// final contour come from the table, scaled by how far the target reaches toward the
    /// emotion's prototype (as Kokoro's style offsets are), the confidence ramp and the
    /// language's intensity. Everything else keeps the rules.
    void set_learned(LearnedProsody learned) { learned_ = std::move(learned); }

    /// Prosody plan for an emotion target (already corrected) and its confidence.
    [[nodiscard]] ProsodyTargets plan(Vad target, float confidence, const ExpressivityProfile& profile) const;

    /// The point to synthesize for a source emotion: source + closed-loop correction.
    [[nodiscard]] Vad target(Vad source) const;

    /// Closed-loop update from 5.2. Only axes the output measurement could see (per-axis
    /// confidence) are corrected. Returns true if the correction changed.
    bool feedback(float ecs, Vad delta, Vad axis_confidence);
    /// Called at each new utterance: the correction decays so it never outlives its cause.
    void next_utterance();

    [[nodiscard]] Vad correction() const noexcept { return correction_; }
    /// Confidence -> [0, 1] scale on every target (smoothstep between floor and full).
    [[nodiscard]] float strength(float confidence) const noexcept;

private:
    ControllerConfig cfg_;
    Vad correction_;
    LearnedProsody learned_;
};

}  // namespace ee
