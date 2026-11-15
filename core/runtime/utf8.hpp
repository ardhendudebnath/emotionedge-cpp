#pragma once
// UTF-8 validity. Whisper decodes byte-level tokens, so a decode that ends or turns mid-character
// leaves half a character in a word. Babble made it hallucinate CJK that way, and the half
// character then broke the session's JSON export.
#include <cstddef>
#include <string>
#include <string_view>

namespace ee {

/// `text` with every invalid UTF-8 sequence dropped: stray continuation bytes, sequences cut
/// short, overlong encodings, surrogates and code points past U+10FFFF. Valid text is unchanged.
[[nodiscard]] inline std::string valid_utf8(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    std::size_t i = 0;
    while (i < text.size()) {
        const auto lead = static_cast<unsigned char>(text[i]);
        if (lead < 0x80) {
            out += text[i++];
            continue;
        }
        std::size_t n = 0;
        char32_t cp = 0;
        char32_t smallest = 0;  // below this the encoding is overlong
        if ((lead >> 5) == 0x6) {
            n = 2, cp = lead & 0x1Fu, smallest = 0x80;
        } else if ((lead >> 4) == 0xE) {
            n = 3, cp = lead & 0x0Fu, smallest = 0x800;
        } else if ((lead >> 3) == 0x1E) {
            n = 4, cp = lead & 0x07u, smallest = 0x10000;
        } else {
            ++i;  // a stray continuation byte, or no lead byte at all
            continue;
        }
        bool whole = i + n <= text.size();
        for (std::size_t k = 1; whole && k < n; ++k) {
            const auto b = static_cast<unsigned char>(text[i + k]);
            whole = (b >> 6) == 0x2;
            cp = (cp << 6) | (b & 0x3Fu);
        }
        if (!whole || cp < smallest || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) {
            ++i;  // drop the lead; its continuation bytes, if any, are dropped as strays
            continue;
        }
        out.append(text.substr(i, n));
        i += n;
    }
    return out;
}

}  // namespace ee
