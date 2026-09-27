#pragma once

#include <array>
#include <cstdint>
#include <filesystem>

#include "core/emotion/emotion_types.hpp"
#include "core/prosody/prosody_types.hpp"

namespace ee {

/// The 128-d style vector sent to StyleTTS2 (blueprint 4.1 "-> style vector (128-d)").
/// Each emotion label has an anchor style (in production: the StyleTTS2 style encoder's mean
/// over reference utterances of that emotion, exported by the ml/ factory). A target point is
/// rendered as a softmax-weighted blend of the anchors by V·A·D proximity, so the vector moves
/// continuously with the emotion rather than jumping between labels.
class StyleBank {
public:
    /// Deterministic unit-norm placeholder anchors until the trained style encoder exports real
    /// ones. Blends are well-defined and testable, but carry no acoustic meaning yet.
    [[nodiscard]] static StyleBank placeholder(std::uint32_t seed = 20240901);
    /// JSON `{"anger": [128 floats], ...}` with one entry per label. Throws ConfigError.
    [[nodiscard]] static StyleBank load(const std::filesystem::path& path);

    [[nodiscard]] StyleVector blend(Vad target, float temperature = 0.35f) const;
    [[nodiscard]] const StyleVector& anchor(EmotionLabel label) const;

private:
    std::array<StyleVector, kEmotionLabels.size()> anchors_{};
};

}  // namespace ee
