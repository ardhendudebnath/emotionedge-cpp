#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <set>
#include <span>
#include <string>
#include <string_view>

#include "core/runtime/params.hpp"

namespace ee {

/// How many target tokens a translation may run to, from its source length (translate params
/// `draft_ratio`, `draft_margin`, `final_ratio`, `final_margin`), and what happens there. A decode
/// that reaches its limit without ending is cut and reported (TranslationResult::truncated), never
/// silently. Measured over FLORES-200 from every source language the runtime maps into Hindi
/// (README, "Decode limits"), the decodes that ran this long were all repetition loops:
/// - a final cut at its limit is decoded once more with a repetition penalty (`retry_penalty`;
///   1 = not), which ended every one of them;
/// - a cut draft (a live caption) is dropped by the stage, so its words are not committed.
/// Drafts get the tighter limit: a looping draft holds every session's batch for nothing.
///
/// Some loops end on their own inside the limit. A decode whose 4-grams of pieces are under
/// `loop_distinct` distinct, over at least `loop_min_pieces` pieces (`loop_distinct` 0: no
/// check), is treated as a cut one (TranslationResult::looped). Translations repeat whole 4-grams
/// rarely; the measured rate is in the README, "Decode limits".
struct DecodeLimits {
    float draft_ratio = 2.5f;
    float draft_margin = 24.0f;
    float final_ratio = 3.0f;
    float final_margin = 32.0f;
    float retry_penalty = 1.2f;
    float loop_distinct = 0.5f;
    std::size_t loop_min_pieces = 12;

    /// True if `pieces` (a decode's output, without its language and end tokens) repeat
    /// themselves as a loop does.
    [[nodiscard]] bool loops(std::span<const std::string> pieces) const {
        if (loop_distinct <= 0.0f || pieces.size() < std::max<std::size_t>(loop_min_pieces, 4)) return false;
        const std::size_t grams = pieces.size() - 3;
        std::set<std::array<std::string_view, 4>> distinct;
        for (std::size_t i = 0; i < grams; ++i) distinct.insert({pieces[i], pieces[i + 1], pieces[i + 2], pieces[i + 3]});
        return static_cast<double>(distinct.size()) < static_cast<double>(loop_distinct) * static_cast<double>(grams);
    }

    /// Target tokens: the target prefix (language token, kept words) and the end token included,
    /// as CTranslate2's max_decoding_length counts them.
    [[nodiscard]] std::size_t limit(bool draft, std::size_t prefix_tokens, std::size_t source_tokens) const noexcept {
        const float ratio = draft ? draft_ratio : final_ratio;
        const float margin = draft ? draft_margin : final_margin;
        return prefix_tokens + static_cast<std::size_t>(std::ceil(ratio * static_cast<float>(source_tokens) + margin));
    }

    [[nodiscard]] static DecodeLimits from(const Params& p) {
        DecodeLimits d;
        d.draft_ratio = p.real("draft_ratio", d.draft_ratio);
        d.draft_margin = p.real("draft_margin", d.draft_margin);
        d.final_ratio = p.real("final_ratio", d.final_ratio);
        d.final_margin = p.real("final_margin", d.final_margin);
        d.retry_penalty = p.real("retry_penalty", d.retry_penalty);
        d.loop_distinct = p.real("loop_distinct", d.loop_distinct);
        d.loop_min_pieces =
            static_cast<std::size_t>(std::max<std::int64_t>(4, p.integer("loop_min_pieces", static_cast<std::int64_t>(d.loop_min_pieces))));
        if (d.draft_ratio <= 0.0f || d.final_ratio <= 0.0f || d.draft_margin < 0.0f || d.final_margin < 0.0f) {
            throw ConfigError("translate decode limits: ratios must be > 0 and margins >= 0");
        }
        if (d.retry_penalty < 1.0f) throw ConfigError("translate retry_penalty must be >= 1 (1: no retry)");
        if (d.loop_distinct < 0.0f || d.loop_distinct > 1.0f) throw ConfigError("translate loop_distinct must be 0-1 (0: no check)");
        return d;
    }
};

}  // namespace ee
