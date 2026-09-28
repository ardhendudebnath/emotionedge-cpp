#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace ee {

struct ChunkerConfig {
    std::size_t min_first_chars = 8;  ///< the first clause may be short: first audio ASAP
    std::size_t min_chars = 24;       ///< later clauses are merged up to at least this
    std::size_t max_chars = 140;      ///< long runs without punctuation are split at a word
    /// Cap on the first clause's words (0 = none). For a TTS whose cost grows with the clause
    /// (Kokoro), the first clause's length sets the time to first audio.
    std::size_t max_first_words = 0;
};

struct Clause {
    std::string text;
    std::vector<std::uint16_t> emphasis;  ///< clause-local word indices
    std::size_t first_word = 0;           ///< index of the clause's first word in the sentence
    std::size_t words = 0;
};

/// Blueprint 4.2 "clause-boundary chunker: first audio before sentence ends". Splits a
/// translation at clause punctuation (Latin, Devanagari danda, CJK and Arabic marks), keeping
/// a short first clause so synthesis can start early, and remaps emphasis indices per clause.
[[nodiscard]] std::vector<Clause> chunk_clauses(std::string_view text, std::string_view language,
                                                const std::vector<std::uint16_t>& emphasis,
                                                const ChunkerConfig& config = {});

/// True if the word ends with clause-ending punctuation.
[[nodiscard]] bool ends_clause(std::string_view word) noexcept;

}  // namespace ee
