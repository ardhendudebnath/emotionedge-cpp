#include "core/tts/text_filter.hpp"

#include <vector>

namespace ee {

namespace {

struct Cp {
    char32_t value;
    std::size_t offset;
    std::size_t size;
};

std::vector<Cp> decode(std::string_view s) {
    std::vector<Cp> out;
    for (std::size_t i = 0; i < s.size();) {
        const auto c = static_cast<unsigned char>(s[i]);
        std::size_t n = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
        if (i + n > s.size()) n = s.size() - i;
        char32_t cp = n == 1 ? c : n == 2 ? (c & 0x1Fu) : n == 3 ? (c & 0x0Fu) : (c & 0x07u);
        for (std::size_t k = 1; k < n; ++k) cp = (cp << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3Fu);
        out.push_back({cp, i, n});
        i += n;
    }
    return out;
}

bool is_symbol(char32_t c) {
    return (c >= 0x2600 && c <= 0x27BF) || (c >= 0x2B00 && c <= 0x2BFF) || (c >= 0xFE00 && c <= 0xFE0F) ||
           (c >= 0x1F000 && c <= 0x1FAFF) || (c >= 0xE0000 && c <= 0xE007F);
}

}  // namespace

std::string remove_emoji(std::string_view text) {
    const std::vector<Cp> cps = decode(text);
    std::string out;
    out.reserve(text.size());
    for (std::size_t i = 0; i < cps.size(); ++i) {
        const char32_t c = cps[i].value;
        if (is_symbol(c)) continue;
        if (c == 0x200D && i > 0 && i + 1 < cps.size() && is_symbol(cps[i - 1].value) && is_symbol(cps[i + 1].value)) {
            continue;  // joins two emoji; a ZWJ inside a word stays
        }
        if (c == ' ' && (out.empty() || out.back() == ' ')) continue;
        out.append(text.substr(cps[i].offset, cps[i].size));
    }
    while (!out.empty() && out.back() == ' ') out.pop_back();
    return out;
}

}  // namespace ee
