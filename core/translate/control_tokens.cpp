#include "core/translate/control_tokens.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include "core/translate/languages.hpp"

namespace ee {

ControlTokens make_control_tokens(const EmotionState& state, Register reg, float tau, float arousal_step) {
    ControlTokens t;
    t.reg = reg;
    if (state.confidence < tau) return t;  // neutral fallback
    t.label = state.label;
    const float step = arousal_step > 0.0f ? arousal_step : 0.01f;
    t.arousal = std::round(std::clamp(state.vad.a, -1.0f, 1.0f) / step) * step;
    return t;
}

std::string format_control_prefix(const ControlTokens& t, float arousal_step) {
    const int decimals = arousal_step >= 0.1f ? 1 : 2;
    char arousal[16];
    std::snprintf(arousal, sizeof arousal, "%.*f", decimals, static_cast<double>(t.arousal == 0.0f ? 0.0f : t.arousal));
    std::string out = "<emo=";
    out += to_string(t.label);
    out += " a=";
    out += arousal;
    out += " reg=";
    out += to_string(t.reg);
    out += '>';
    return out;
}

std::string_view strip_control_prefix(std::string_view text) noexcept {
    if (text.substr(0, 5) != "<emo=") return text;
    const auto end = text.find('>');
    if (end == std::string_view::npos) return text;
    text.remove_prefix(end + 1);
    while (!text.empty() && text.front() == ' ') text.remove_prefix(1);
    return text;
}

std::string remove_control_tokens(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    std::size_t i = 0;
    while (i < text.size()) {
        const std::size_t start = text.find("<emo=", i);
        const std::size_t end = start == std::string_view::npos ? start : text.find('>', start);
        if (end == std::string_view::npos) {
            out.append(text.substr(i));
            break;
        }
        out.append(text.substr(i, start - i));
        i = end + 1;
    }
    // Collapse the doubled or edge spaces a removed span leaves behind.
    std::string clean;
    clean.reserve(out.size());
    for (char c : out) {
        if (c == ' ' && (clean.empty() || clean.back() == ' ')) continue;
        clean += c;
    }
    while (!clean.empty() && clean.back() == ' ') clean.pop_back();
    return clean;
}

std::string apply_emphasis_markup(const std::vector<std::string>& words, const std::vector<std::uint16_t>& emphasis,
                                  std::string_view language) {
    std::vector<std::string> marked = words;
    for (std::uint16_t i : emphasis) {
        if (i < marked.size()) marked[i] = "<em>" + marked[i] + "</em>";
    }
    return join_text(marked, language);
}

Markup parse_markup(std::string_view text, std::string_view language) {
    // Rebuild the plain text while tracking which byte ranges were inside <em>...</em>.
    Markup out;
    std::vector<std::pair<std::size_t, std::size_t>> spans;  // byte ranges in `plain`
    std::size_t open_at = std::string::npos;
    for (std::size_t i = 0; i < text.size();) {
        if (text.compare(i, 4, "<em>") == 0) {
            open_at = out.plain.size();
            i += 4;
        } else if (text.compare(i, 5, "</em>") == 0) {
            if (open_at != std::string::npos) spans.push_back({open_at, out.plain.size()});
            open_at = std::string::npos;
            i += 5;
        } else {
            out.plain += text[i];
            ++i;
        }
    }
    if (spans.empty()) return out;

    // Map byte spans to word indices of split_words(plain).
    const std::vector<std::string> words = split_words(out.plain, language);
    std::size_t cursor = 0;
    for (std::size_t w = 0; w < words.size(); ++w) {
        const std::size_t begin = out.plain.find(words[w], cursor);
        if (begin == std::string::npos) break;
        const std::size_t end = begin + words[w].size();
        cursor = end;
        const bool emphasized = std::any_of(spans.begin(), spans.end(), [&](const auto& s) {
            return begin < s.second && s.first < end;  // overlaps
        });
        if (emphasized) out.emphasis.push_back(static_cast<std::uint16_t>(w));
    }
    return out;
}

}  // namespace ee
