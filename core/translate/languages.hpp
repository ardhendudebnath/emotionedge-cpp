#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ee {

/// NLLB-200 / FLORES-200 code for an ISO 639-1 language ("hi" -> "hin_Deva").
[[nodiscard]] std::optional<std::string_view> nllb_code(std::string_view iso639_1) noexcept;

/// Whether the language separates words with spaces (so words can be split on whitespace).
[[nodiscard]] bool uses_spaces(std::string_view iso639_1) noexcept;

/// Splits text into words: on whitespace, or per character for scripts written without
/// spaces (Chinese, Japanese, Thai, ...), so emphasis indices always have something to point at.
[[nodiscard]] std::vector<std::string> split_words(std::string_view text, std::string_view language = "en");

/// Joins words back with the language's separator.
[[nodiscard]] std::string join_text(const std::vector<std::string>& words, std::string_view language = "en");

}  // namespace ee
