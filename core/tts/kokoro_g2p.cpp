#include "core/tts/kokoro_g2p.hpp"

#include <algorithm>
#include <array>
#include <utility>

namespace ee::kokoro {

namespace {

struct CodePoint {
    char32_t value;
    std::size_t offset;
};

std::vector<CodePoint> decode(std::string_view s) {
    std::vector<CodePoint> out;
    for (std::size_t i = 0; i < s.size();) {
        const auto c = static_cast<unsigned char>(s[i]);
        const std::size_t n = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
        char32_t cp = n == 1 ? c : n == 2 ? (c & 0x1Fu) : n == 3 ? (c & 0x0Fu) : (c & 0x07u);
        for (std::size_t k = 1; k < n && i + k < s.size(); ++k) cp = (cp << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3Fu);
        out.push_back({cp, i});
        i += n;
    }
    return out;
}

// Python's str.isspace / regex \s.
bool is_space(char32_t c) {
    return c == ' ' || (c >= 0x09 && c <= 0x0D) || (c >= 0x1C && c <= 0x1F) || c == 0x85 || c == 0xA0 ||
           c == 0x1680 || (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F ||
           c == 0x205F || c == 0x3000;
}

// phonemizer's default marks: ;:,.!?¡¿—…"«»“”(){}[]
bool is_mark(char32_t c) {
    static constexpr std::array<char32_t, 21> kMarks = {U';', U':', U',', U'.', U'!', U'?', U'¡',
                                                        U'¿', U'—', U'…', U'"', U'«', U'»', U'“',
                                                        U'”', U'(', U')', U'{', U'}', U'[', U']'};
    return std::find(kMarks.begin(), kMarks.end(), c) != kMarks.end();
}

std::string strip(std::string_view s) {
    const std::vector<CodePoint> cps = decode(s);
    std::size_t begin = 0;
    std::size_t end = cps.size();
    while (begin < end && is_space(cps[begin].value)) ++begin;
    while (end > begin && is_space(cps[end - 1].value)) --end;
    if (begin == end) return {};
    const std::size_t from = cps[begin].offset;
    const std::size_t to = end < cps.size() ? cps[end].offset : s.size();
    return std::string(s.substr(from, to - from));
}

void replace_all(std::string& s, std::string_view from, std::string_view to) {
    if (from.empty()) return;
    std::string out;
    out.reserve(s.size());
    std::size_t i = 0;
    while (true) {
        const std::size_t hit = s.find(from, i);
        if (hit == std::string::npos) break;
        out.append(s, i, hit - i).append(to);
        i = hit + from.size();
    }
    out.append(s, i, std::string::npos);
    s = std::move(out);
}

bool ends_with(const std::string& s, std::string_view tail) {
    return s.size() >= tail.size() && s.compare(s.size() - tail.size(), tail.size(), tail) == 0;
}

}  // namespace

PunctuationSplit preserve_punctuation(std::string_view line) {
    PunctuationSplit out;
    // Matches of (\s*[marks]+\s*)+ are the maximal runs of spaces and marks holding a mark.
    const std::vector<CodePoint> cps = decode(line);
    std::vector<std::pair<std::size_t, std::size_t>> matches;  // byte ranges
    for (std::size_t i = 0; i < cps.size();) {
        if (!is_space(cps[i].value) && !is_mark(cps[i].value)) {
            ++i;
            continue;
        }
        std::size_t j = i;
        bool has_mark = false;
        while (j < cps.size() && (is_space(cps[j].value) || is_mark(cps[j].value))) {
            has_mark = has_mark || is_mark(cps[j].value);
            ++j;
        }
        if (has_mark) matches.push_back({cps[i].offset, j < cps.size() ? cps[j].offset : line.size()});
        i = j;
    }
    if (matches.empty()) {
        if (!line.empty()) out.chunks.emplace_back(line);
        return out;
    }
    if (matches.size() == 1 && matches[0].first == 0 && matches[0].second == line.size()) {
        out.marks.push_back({std::string(line), 'A'});
        return out;
    }
    for (std::size_t m = 0; m < matches.size(); ++m) {
        const std::string text(line.substr(matches[m].first, matches[m].second - matches[m].first));
        char position = 'I';
        if (m == 0 && line.substr(0, text.size()) == text) {
            position = 'B';
        } else if (m + 1 == matches.size() && line.size() >= text.size() &&
                   line.substr(line.size() - text.size()) == text) {
            position = 'E';
        }
        out.marks.push_back({text, position});
    }
    // Split at the first occurrence of each mark in turn, like str.split + join.
    std::string rest(line);
    for (const PunctuationMark& mark : out.marks) {
        const std::size_t at = rest.find(mark.text);
        if (at == std::string::npos) break;
        out.chunks.push_back(rest.substr(0, at));
        rest.erase(0, at + mark.text.size());
    }
    out.chunks.push_back(rest);
    std::erase_if(out.chunks, [](const std::string& c) { return c.empty(); });
    return out;
}

std::string restore_punctuation(std::vector<std::string> text, const std::vector<PunctuationMark>& marks) {
    std::vector<std::string> out;
    std::size_t next = 0;  // marks consumed
    int pos = 0;           // every mark of a single line has index 0
    while (!text.empty() || next < marks.size()) {
        if (next == marks.size()) {
            for (std::string& line : text) {
                if (!ends_with(line, " ")) line += ' ';
                out.push_back(line);
            }
            text.clear();
        } else if (text.empty()) {
            std::string joined;
            for (; next < marks.size(); ++next) joined += marks[next].text;
            out.push_back(joined);
        } else if (pos == 0) {
            const PunctuationMark& mark = marks[next++];
            if (ends_with(text[0], " ")) text[0].pop_back();
            const std::string after = ends_with(mark.text, " ") ? "" : " ";
            if (mark.position == 'B') {
                text[0] = mark.text + text[0];
            } else if (mark.position == 'E') {
                out.push_back(text[0] + mark.text + after);
                text.erase(text.begin());
                ++pos;
            } else if (mark.position == 'A') {
                out.push_back(mark.text + after);
                ++pos;
            } else if (text.size() == 1) {
                text[0] += mark.text;
            } else {
                const std::string first = text[0];
                text.erase(text.begin());
                text[0] = first + mark.text + text[0];
            }
        } else {
            out.push_back(text[0]);
            text.erase(text.begin());
            ++pos;
        }
    }
    return out.empty() ? std::string() : out.front();  // misaki reads the first line only
}

std::string postprocess_espeak_line(std::string_view raw) {
    std::string line = strip(raw);
    replace_all(line, "\n", " ");
    replace_all(line, "  ", " ");  // one pass, like str.replace
    // re.sub('_+', '_') then re.sub('_ ', ' '): espeak can leave separators after words.
    std::string merged;
    for (char c : line) {
        if (c == '_' && !merged.empty() && merged.back() == '_') continue;
        merged += c;
    }
    replace_all(merged, "_ ", " ");
    line = std::move(merged);
    // Language-switch flags such as "(en)" ... "(hi)": remove every \(.+?\).
    std::string unflagged;
    for (std::size_t i = 0; i < line.size();) {
        if (line[i] == '(') {
            const std::size_t close = line.find(')', i + 2);
            if (close != std::string::npos) {
                i = close + 1;
                continue;
            }
        }
        unflagged += line[i++];
    }
    line = std::move(unflagged);
    if (line.empty()) return {};

    std::string out;
    std::size_t start = 0;
    while (true) {
        const std::size_t space = line.find(' ', start);
        std::string word = strip(std::string_view(line).substr(start, space == std::string::npos ? std::string::npos
                                                                                                 : space - start));
        replace_all(word, "\xCD\xA1", "^");  // U+0361 COMBINING DOUBLE INVERTED BREVE -> '^'
        out += word;
        out += ' ';
        if (space == std::string::npos) break;
        start = space + 1;
    }
    return out;
}

std::string misaki_g2p(std::string_view text, const std::function<std::string(const std::string&)>& espeak) {
    std::string t(text);
    replace_all(t, "«", "“");
    replace_all(t, "»", "”");
    replace_all(t, "(", "«");
    replace_all(t, ")", "»");

    const PunctuationSplit split = preserve_punctuation(t);
    std::vector<std::string> phonemized;
    phonemized.reserve(split.chunks.size());
    for (const std::string& chunk : split.chunks) phonemized.push_back(postprocess_espeak_line(espeak(chunk)));
    std::string ps = strip(restore_punctuation(std::move(phonemized), split.marks));

    // Tied pairs -> Kokoro's single-letter symbols, in misaki's (sorted) order.
    static const std::array<std::pair<std::string_view, std::string_view>, 11> kMerges = {{
        {"a^ɪ", "I"}, {"a^ʊ", "W"}, {"d^z", "ʣ"}, {"d^ʒ", "ʤ"}, {"e^ɪ", "A"}, {"o^ʊ", "O"},
        {"s^s", "S"}, {"t^s", "ʦ"}, {"t^ʃ", "ʧ"}, {"ɔ^ɪ", "Y"}, {"ə^ʊ", "Q"},
    }};
    for (const auto& [from, to] : kMerges) replace_all(ps, from, to);
    replace_all(ps, "^", "");
    replace_all(ps, "-", "");
    replace_all(ps, "«", "(");
    replace_all(ps, "»", ")");
    return ps;
}

}  // namespace ee::kokoro
