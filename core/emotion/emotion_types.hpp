#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string_view>

namespace ee {

/// A point in the Valence · Arousal · Dominance space that every stage speaks (blueprint p.3),
/// so the emotion measured on the input can be measured again on the output. Axes span [-1, 1].
struct Vad {
    float v = 0.0f;  ///< valence: unpleasant (-) ... pleasant (+)
    float a = 0.0f;  ///< arousal: calm (-) ... excited (+)
    float d = 0.0f;  ///< dominance: submissive (-) ... in control (+)

    friend constexpr Vad operator+(Vad x, Vad y) { return {x.v + y.v, x.a + y.a, x.d + y.d}; }
    friend constexpr Vad operator-(Vad x, Vad y) { return {x.v - y.v, x.a - y.a, x.d - y.d}; }
    friend constexpr Vad operator*(Vad x, float s) { return {x.v * s, x.a * s, x.d * s}; }
    friend constexpr Vad operator*(float s, Vad x) { return x * s; }
    friend constexpr bool operator==(const Vad&, const Vad&) = default;

    [[nodiscard]] float norm() const { return std::sqrt(v * v + a * a + d * d); }
    [[nodiscard]] constexpr Vad clamped(float lo = -1.0f, float hi = 1.0f) const {
        return {std::clamp(v, lo, hi), std::clamp(a, lo, hi), std::clamp(d, lo, hi)};
    }
};

[[nodiscard]] inline float distance(Vad x, Vad y) { return (x - y).norm(); }
[[nodiscard]] constexpr Vad lerp(Vad x, Vad y, float t) { return x + (y - x) * t; }

/// Diameter of the [-1, 1]^3 cube (2·sqrt(3)): the largest possible V·A·D distance.
inline constexpr float kVadDiameter = 3.4641016151377544f;

/// Discrete labels used for control tokens and captions. The continuous V·A·D point is what
/// the pipeline actually carries; the label is the nearest prototype, for humans and MT tokens.
enum class EmotionLabel : std::uint8_t { Neutral, Joy, Surprise, Anger, Fear, Sadness, Calm };

inline constexpr std::array<EmotionLabel, 7> kEmotionLabels = {
    EmotionLabel::Neutral, EmotionLabel::Joy,     EmotionLabel::Surprise, EmotionLabel::Anger,
    EmotionLabel::Fear,    EmotionLabel::Sadness, EmotionLabel::Calm};

[[nodiscard]] constexpr std::string_view to_string(EmotionLabel label) noexcept {
    switch (label) {
    case EmotionLabel::Neutral: return "neutral";
    case EmotionLabel::Joy: return "joy";
    case EmotionLabel::Surprise: return "surprise";
    case EmotionLabel::Anger: return "anger";
    case EmotionLabel::Fear: return "fear";
    case EmotionLabel::Sadness: return "sadness";
    case EmotionLabel::Calm: return "calm";
    }
    return "neutral";
}

[[nodiscard]] constexpr std::optional<EmotionLabel> parse_emotion_label(std::string_view s) noexcept {
    for (EmotionLabel label : kEmotionLabels) {
        if (to_string(label) == s) return label;
    }
    return std::nullopt;
}

/// Canonical position of each label, read off the blueprint's emotion map (p.3). Anger and fear
/// sit close on V×A; dominance (D+ vs D−) is what keeps a furious voice from being
/// re-synthesized as a frightened one. Mirrored in config/emotion_space.json for the ml/ factory.
[[nodiscard]] constexpr Vad prototype(EmotionLabel label) noexcept {
    switch (label) {
    case EmotionLabel::Neutral: return {0.00f, 0.00f, 0.00f};
    case EmotionLabel::Joy: return {0.75f, 0.50f, 0.35f};
    case EmotionLabel::Surprise: return {0.30f, 0.80f, -0.10f};
    case EmotionLabel::Anger: return {-0.62f, 0.78f, 0.55f};
    case EmotionLabel::Fear: return {-0.45f, 0.50f, -0.55f};
    case EmotionLabel::Sadness: return {-0.65f, -0.50f, -0.35f};
    case EmotionLabel::Calm: return {0.55f, -0.45f, 0.15f};
    }
    return {};
}

/// Label whose prototype is closest to `p`.
[[nodiscard]] inline EmotionLabel nearest_label(Vad p) noexcept {
    EmotionLabel best = EmotionLabel::Neutral;
    float best_distance = std::numeric_limits<float>::max();
    for (EmotionLabel label : kEmotionLabels) {
        const float dist = distance(p, prototype(label));
        if (dist < best_distance) {
            best_distance = dist;
            best = label;
        }
    }
    return best;
}

/// Blueprint 3.1 output: `EmotionState{V, A, D, conf}` plus the display label.
struct EmotionState {
    Vad vad;
    EmotionLabel label = EmotionLabel::Neutral;
    float confidence = 0.0f;  ///< [0, 1]
};

}  // namespace ee
