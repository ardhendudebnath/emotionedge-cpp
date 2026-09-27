#include "core/tts/clause_chunker.hpp"

#include <algorithm>
#include <array>

#include "core/translate/languages.hpp"

namespace ee {

bool ends_clause(std::string_view word) noexcept {
    static constexpr std::array<std::string_view, 17> kMarks = {
        ",", ";", ":", ".", "!", "?", "\xE2\x80\xA6" /* … */, "\xE0\xA5\xA4" /* । */, "\xE0\xA5\xA5" /* ॥ */,
        "\xE3\x80\x81" /* 、 */, "\xE3\x80\x82" /* 。 */, "\xEF\xBC\x8C" /* ， */, "\xEF\xBC\x81" /* ！ */,
        "\xEF\xBC\x9F" /* ？ */, "\xD8\x8C" /* ، */, "\xD8\x9F" /* ؟ */, "\xEF\xBC\x9B" /* ； */};
    // Ignore closing quotes/brackets after the mark: `said," she`.
    while (!word.empty() && (word.back() == '"' || word.back() == '\'' || word.back() == ')')) word.remove_suffix(1);
    return std::any_of(kMarks.begin(), kMarks.end(),
                       [&](std::string_view m) { return word.size() >= m.size() && word.substr(word.size() - m.size()) == m; });
}

std::vector<Clause> chunk_clauses(std::string_view text, std::string_view language,
                                  const std::vector<std::uint16_t>& emphasis, const ChunkerConfig& cfg) {
    const std::vector<std::string> words = split_words(text, language);
    const std::string_view sep = uses_spaces(language) ? " " : "";
    std::vector<Clause> out;
    Clause current;
    const auto flush = [&] {
        if (current.words == 0) return;
        out.push_back(std::move(current));
        current = Clause{};
    };
    for (std::size_t i = 0; i < words.size(); ++i) {
        if (current.words == 0) current.first_word = i;
        if (!current.text.empty()) current.text += sep;
        current.text += words[i];
        if (std::find(emphasis.begin(), emphasis.end(), static_cast<std::uint16_t>(i)) != emphasis.end()) {
            current.emphasis.push_back(static_cast<std::uint16_t>(current.words));
        }
        ++current.words;
        const std::size_t min_len = out.empty() ? cfg.min_first_chars : cfg.min_chars;
        const bool last = i + 1 == words.size();
        if (!last && ((ends_clause(words[i]) && current.text.size() >= min_len) || current.text.size() >= cfg.max_chars)) {
            flush();
        }
    }
    // A very short tail is merged into the clause before it.
    if (!out.empty() && current.words > 0 && current.text.size() < cfg.min_chars / 2) {
        Clause& prev = out.back();
        for (std::uint16_t e : current.emphasis) prev.emphasis.push_back(static_cast<std::uint16_t>(prev.words + e));
        prev.text += sep;
        prev.text += current.text;
        prev.words += current.words;
        current = Clause{};
    }
    flush();
    return out;
}

}  // namespace ee
