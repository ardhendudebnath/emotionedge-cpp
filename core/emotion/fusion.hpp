#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "core/asr/transcript.hpp"
#include "core/emotion/acoustic.hpp"
#include "core/runtime/params.hpp"

namespace ee {

/// Blueprint 2.2 "gated late fusion (confidence-weighted)".
struct FusionConfig {
    /// Prior trust in each modality per axis (valence, arousal, dominance): prosody is strong on
    /// arousal, words are strong on valence.
    Vad acoustic_weight{0.3f, 1.0f, 0.8f};
    Vad lexical_weight{1.0f, 0.5f, 0.4f};
    float gate = 0.1f;  ///< an axis whose confidence is below this is ignored for that modality

    static FusionConfig from(const Params& params);
};

/// Fuses per axis: value = Σ w·c·x / Σ w·c over the modalities that pass the gate, and
/// confidence = 1 - Π(1 - c), since independent evidence accumulates. The overall confidence is
/// the mean over the three axes (also returned per axis if requested); the label is the
/// nearest prototype.
[[nodiscard]] EmotionState fuse(const ModalityEstimate& acoustic, const ModalityEstimate& lexical,
                                const FusionConfig& config, Vad* axis_confidence = nullptr);

/// "label + confidence, EMA-smoothed": confident estimates move the state faster.
class EmotionSmoother {
public:
    explicit EmotionSmoother(float alpha = 0.5f) : alpha_(alpha) {}
    /// Blends `next` into the running state with weight alpha · confidence (the first estimate
    /// is taken as-is) and returns the smoothed state.
    EmotionState update(const EmotionState& next, float alpha_override = -1.0f);
    [[nodiscard]] const EmotionState& state() const noexcept { return state_; }
    void reset() { initialized_ = false; }

private:
    float alpha_;
    bool initialized_ = false;
    EmotionState state_;
};

/// Blueprint 3.1 "hysteresis: no label flicker". The displayed label changes when a new label
/// wins by `strong_margin` at once, or by `margin` for `updates` consecutive updates.
class LabelHysteresis {
public:
    LabelHysteresis(float margin = 0.08f, float strong_margin = 0.25f, int updates = 2)
        : margin_(margin), strong_margin_(strong_margin), updates_(updates) {}
    EmotionLabel update(Vad point);
    [[nodiscard]] EmotionLabel current() const noexcept { return current_; }

private:
    float margin_;
    float strong_margin_;
    int updates_;
    bool initialized_ = false;
    EmotionLabel current_ = EmotionLabel::Neutral;
    EmotionLabel candidate_ = EmotionLabel::Neutral;
    int streak_ = 0;
};

struct EmphasisConfig {
    float z_threshold = 1.0f;   ///< detrended word peak energy z-score across the utterance
    float min_rise_db = 2.5f;   ///< and at least this far above the median word peak
    int max_words = 2;
    float min_word_s = 0.06f;
};

/// Blueprint 3.1 "emphasis = energy peaks aligned to ASR word timestamps": indices of words
/// whose peak energy stands out from the rest of the utterance.
[[nodiscard]] std::vector<std::uint16_t> find_emphasis(const std::vector<Word>& words,
                                                       std::span<const float> envelope_db, float hop_s,
                                                       const EmphasisConfig& config = {});

}  // namespace ee
