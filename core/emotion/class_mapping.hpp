#pragma once

#include <cstddef>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include "core/emotion/acoustic.hpp"

namespace ee {

/// Turns a categorical emotion classifier into one modality's V·A·D opinion (2.2). Both phase-2
/// models are classifiers: emotion2vec+ (9 classes) and DistilRoBERTa (7 classes).
///
/// The model's class probabilities p are handled as follows:
///  - "abstain" classes (emotion2vec's "other"/"unknown") carry no evidence. Their mass q is set
///    aside and the rest renormalized to p'.
///  - V·A·D = Σ p'_k · vad_k, the probability-weighted class position.
///  - strength = 1 − p'_neutral: neutral words are weak evidence of a neutral speaker.
///  - certainty = 1 − H(p') / ln K, where K counts the non-abstain classes.
///  - per-axis confidence = reliability · (0.3 + 0.7·strength) · (0.5 + 0.5·certainty) · (1 − q).
///
/// Class positions and per-axis reliability come from the model's `labels.json`, written by
/// the ml/ export scripts from config/emotion_space.json.
struct ClassEmotionMap {
    std::vector<std::string> labels;
    std::vector<Vad> vad;        ///< position of each class
    std::vector<bool> abstain;   ///< classes without emotion evidence
    std::size_t neutral = kNone; ///< index of the neutral class
    Vad reliability{1.0f, 1.0f, 1.0f};

    static constexpr std::size_t kNone = static_cast<std::size_t>(-1);

    /// Reads `{"labels": [...], "vad": {label: [v, a, d]}, "abstain": [...], "neutral": "...",
    /// "reliability": [v, a, d]}`. Every non-abstain label needs a position. Throws ConfigError.
    [[nodiscard]] static ClassEmotionMap from_file(const std::filesystem::path& labels_json);

    [[nodiscard]] ModalityEstimate from_probabilities(std::span<const float> probabilities) const;
    [[nodiscard]] ModalityEstimate from_logits(std::span<const float> logits) const;
};

}  // namespace ee
