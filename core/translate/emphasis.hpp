#pragma once

#include <cstdint>
#include <utility>
#include <vector>

namespace ee {

/// Blueprint 3.2 "emphasis spans -> target words". Carries emphasized source words over to the
/// target sentence using the best evidence available:
///  1. `<em>` markup that survived translation (`markup`, already parsed),
///  2. the engine's word alignment (attention),
///  3. relative position in the sentence (fallback).
[[nodiscard]] std::vector<std::uint16_t> project_emphasis(
    const std::vector<std::uint16_t>& source_emphasis, std::size_t source_words, std::size_t target_words,
    const std::vector<std::uint16_t>& markup,
    const std::vector<std::pair<std::uint16_t, std::uint16_t>>& alignment);

}  // namespace ee
