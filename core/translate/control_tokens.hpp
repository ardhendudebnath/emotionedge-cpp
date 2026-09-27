#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "core/emotion/emotion_types.hpp"
#include "core/prosody/expressivity.hpp"

namespace ee {

/// The emotion control prefix for the MT model (blueprint 3.2):
/// `<emo=anger a=0.78 reg=casual>`.
struct ControlTokens {
    EmotionLabel label = EmotionLabel::Neutral;
    float arousal = 0.0f;
    Register reg = Register::Casual;
};

/// Builds the tokens for an emotion state, falling back to neutral when confidence < `tau`
/// ("neutral fallback if conf < τ"). Arousal is quantized to `arousal_step`.
[[nodiscard]] ControlTokens make_control_tokens(const EmotionState& state, Register reg, float tau,
                                                float arousal_step = 0.01f);

[[nodiscard]] std::string format_control_prefix(const ControlTokens& tokens, float arousal_step = 0.01f);

/// Removes a leading `<emo=...>` prefix (and the space after it), if present.
[[nodiscard]] std::string_view strip_control_prefix(std::string_view text) noexcept;

/// Joins words, wrapping the emphasized ones: `I can't <em>believe</em> you did this!`.
[[nodiscard]] std::string apply_emphasis_markup(const std::vector<std::string>& words,
                                                const std::vector<std::uint16_t>& emphasis,
                                                std::string_view language = "en");

struct Markup {
    std::string plain;                  ///< text without <em> tags
    std::vector<std::uint16_t> emphasis;  ///< indices of words that were inside <em>...</em>
};

/// Strips `<em>` markup, reporting which words (of split_words(plain)) were emphasized.
[[nodiscard]] Markup parse_markup(std::string_view text, std::string_view language = "en");

}  // namespace ee
