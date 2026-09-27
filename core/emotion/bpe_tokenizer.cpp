#include "core/emotion/bpe_tokenizer.hpp"

#include <nlohmann/json.hpp>

#include <array>
#include <climits>
#include <fstream>

#include "core/runtime/params.hpp"

namespace ee {

namespace {

// ---- UTF-8 ---------------------------------------------------------------------------------

struct CodePoint {
    char32_t value;
    std::size_t offset;  ///< byte offset in the text
    std::size_t size;    ///< byte length
};

std::vector<CodePoint> decode_utf8(std::string_view s) {
    std::vector<CodePoint> out;
    out.reserve(s.size());
    std::size_t i = 0;
    while (i < s.size()) {
        const auto b0 = static_cast<unsigned char>(s[i]);
        std::size_t n = 1;
        char32_t cp = b0;
        if (b0 >= 0xF0 && b0 < 0xF8) {
            n = 4;
            cp = b0 & 0x07u;
        } else if (b0 >= 0xE0) {
            n = 3;
            cp = b0 & 0x0Fu;
        } else if (b0 >= 0xC0) {
            n = 2;
            cp = b0 & 0x1Fu;
        }
        bool valid = i + n <= s.size() && b0 < 0xF8 && (b0 < 0x80 || b0 >= 0xC0);
        for (std::size_t k = 1; valid && k < n; ++k) {
            const auto b = static_cast<unsigned char>(s[i + k]);
            if ((b & 0xC0u) != 0x80u) valid = false;
            cp = (cp << 6) | (b & 0x3Fu);
        }
        if (!valid) {  // malformed: take the byte on its own; byte-level BPE still encodes it
            n = 1;
            cp = 0xFFFD;
        }
        out.push_back({cp, i, n});
        i += n;
    }
    return out;
}

void append_utf8(std::string& out, char32_t cp) {
    if (cp < 0x80) {
        out += static_cast<char>(cp);
    } else if (cp < 0x800) {
        out += static_cast<char>(0xC0 | (cp >> 6));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        out += static_cast<char>(0xE0 | (cp >> 12));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        out += static_cast<char>(0xF0 | (cp >> 18));
        out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    }
}

// ---- GPT-2 character classes ------------------------------------------------------------------

enum class CharClass { Letter, Number, Space, Other };

bool in(char32_t c, char32_t lo, char32_t hi) { return c >= lo && c <= hi; }

CharClass classify(char32_t c) {
    if (c < 0x80) {
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) return CharClass::Letter;
        if (c >= '0' && c <= '9') return CharClass::Number;
        if (c == ' ' || (c >= 0x09 && c <= 0x0D)) return CharClass::Space;
        return CharClass::Other;
    }
    // \s beyond ASCII.
    if (c == 0x85 || c == 0xA0 || c == 0x1680 || in(c, 0x2000, 0x200A) || c == 0x2028 || c == 0x2029 ||
        c == 0x202F || c == 0x205F || c == 0x3000) {
        return CharClass::Space;
    }
    // Latin-1 supplement: exact.
    if (c < 0x100) {
        if (c == 0xAA || c == 0xB5 || c == 0xBA) return CharClass::Letter;
        if (c == 0xB2 || c == 0xB3 || c == 0xB9 || in(c, 0xBC, 0xBE)) return CharClass::Number;
        if (c < 0xC0 || c == 0xD7 || c == 0xF7) return CharClass::Other;
        return CharClass::Letter;
    }
    // Decimal digits of common scripts, and fullwidth digits.
    static constexpr std::array<char32_t, 14> kDigitBlocks = {0x0660, 0x06F0, 0x0966, 0x09E6, 0x0A66, 0x0AE6, 0x0B66,
                                                               0x0BE6, 0x0C66, 0x0CE6, 0x0D66, 0x0E50, 0x0ED0, 0xFF10};
    for (char32_t base : kDigitBlocks) {
        if (in(c, base, base + 9)) return CharClass::Number;
    }
    // Combining marks, punctuation, symbols, private use and emoji.
    if (in(c, 0x0300, 0x036F) || in(c, 0x2010, 0x2027) || in(c, 0x2030, 0x205E) || in(c, 0x20A0, 0x20FF) ||
        in(c, 0x2100, 0x214F) || in(c, 0x2190, 0x2BFF) || in(c, 0x3001, 0x303F) || in(c, 0xE000, 0xF8FF) ||
        in(c, 0xFE30, 0xFE4F) || in(c, 0xFF01, 0xFF0F) || in(c, 0xFF1A, 0xFF20) || in(c, 0xFF3B, 0xFF40) ||
        in(c, 0xFF5B, 0xFF65) || c == 0xFFFD || in(c, 0x1F000, 0x1FAFF)) {
        return CharClass::Other;
    }
    return CharClass::Letter;
}

// ---- GPT-2 byte <-> printable character table ---------------------------------------------

std::array<std::string, 256> byte_to_symbol_table() {
    std::array<std::string, 256> table;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        const bool printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
        const char32_t cp = printable ? static_cast<char32_t>(b) : static_cast<char32_t>(256 + n++);
        append_utf8(table[static_cast<std::size_t>(b)], cp);
    }
    return table;
}

const std::array<std::string, 256>& byte_symbols() {
    static const std::array<std::string, 256> table = byte_to_symbol_table();
    return table;
}

}  // namespace

ByteLevelBpeTokenizer::ByteLevelBpeTokenizer(std::unordered_map<std::string, std::int64_t> vocab,
                                             const std::vector<std::pair<std::string, std::string>>& merges,
                                             std::int64_t bos_id, std::int64_t eos_id, bool add_prefix_space)
    : vocab_(std::move(vocab)), bos_id_(bos_id), eos_id_(eos_id), add_prefix_space_(add_prefix_space) {
    merge_rank_.reserve(merges.size());
    for (std::size_t i = 0; i < merges.size(); ++i) {
        // Byte-mapped symbols never contain a plain space (0x20 maps to "Ġ"), so it is a safe separator.
        merge_rank_.emplace(merges[i].first + ' ' + merges[i].second, static_cast<int>(i));
    }
}

ByteLevelBpeTokenizer ByteLevelBpeTokenizer::from_tokenizer_json(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw ConfigError("cannot open tokenizer '" + path.string() + "'");
    nlohmann::json j;
    try {
        in >> j;
    } catch (const nlohmann::json::exception& e) {
        throw ConfigError("tokenizer '" + path.string() + "': " + e.what());
    }
    const auto& model = j.at("model");
    if (model.value("type", std::string("BPE")) != "BPE") throw ConfigError("tokenizer '" + path.string() + "' is not BPE");

    std::unordered_map<std::string, std::int64_t> vocab;
    for (const auto& [token, id] : model.at("vocab").items()) vocab.emplace(token, id.get<std::int64_t>());
    std::vector<std::pair<std::string, std::string>> merges;
    for (const auto& m : model.at("merges")) {
        if (m.is_array()) {  // newer format: ["a", "b"]
            merges.emplace_back(m.at(0).get<std::string>(), m.at(1).get<std::string>());
        } else {  // "a b"
            const std::string s = m.get<std::string>();
            const auto space = s.find(' ');
            if (space == std::string::npos) throw ConfigError("tokenizer '" + path.string() + "': bad merge '" + s + "'");
            merges.emplace_back(s.substr(0, space), s.substr(space + 1));
        }
    }

    bool add_prefix_space = false;
    if (const auto pre = j.find("pre_tokenizer"); pre != j.end() && pre->is_object()) {
        if (pre->value("type", std::string()) != "ByteLevel") {
            throw ConfigError("tokenizer '" + path.string() + "': expected a ByteLevel pre-tokenizer");
        }
        add_prefix_space = pre->value("add_prefix_space", false);
    }
    const auto special = [&](const char* role, const char* fallback) {
        if (const auto post = j.find("post_processor"); post != j.end() && post->is_object() && post->contains(role)) {
            return post->at(role).at(1).get<std::int64_t>();  // RobertaProcessing: [token, id]
        }
        const auto it = vocab.find(fallback);
        if (it == vocab.end()) throw ConfigError("tokenizer '" + path.string() + "' has no " + fallback);
        return it->second;
    };
    const std::int64_t bos = special("cls", "<s>");
    const std::int64_t eos = special("sep", "</s>");
    return ByteLevelBpeTokenizer(std::move(vocab), merges, bos, eos, add_prefix_space);
}

std::vector<std::string> ByteLevelBpeTokenizer::pre_tokenize(std::string_view text) const {
    std::string prefixed;
    if (add_prefix_space_ && !text.empty() && text.front() != ' ') {
        prefixed = " " + std::string(text);
        text = prefixed;
    }
    const std::vector<CodePoint> cps = decode_utf8(text);
    const std::size_t n = cps.size();
    std::vector<std::string> pieces;
    const auto emit = [&](std::size_t from, std::size_t to) {  // code point range [from, to)
        const std::size_t begin = cps[from].offset;
        const std::size_t end = to < n ? cps[to].offset : text.size();
        pieces.emplace_back(text.substr(begin, end - begin));
    };
    const auto run_end = [&](std::size_t from, CharClass cls) {
        while (from < n && classify(cps[from].value) == cls) ++from;
        return from;
    };

    std::size_t i = 0;
    while (i < n) {
        const char32_t c = cps[i].value;
        // 's 't 're 've 'm 'll 'd (case-sensitive, like the pattern).
        if (c == U'\'' && i + 1 < n) {
            const char32_t c1 = cps[i + 1].value;
            const char32_t c2 = i + 2 < n ? cps[i + 2].value : 0;
            std::size_t len = 0;
            if (c1 == U's' || c1 == U't' || c1 == U'm' || c1 == U'd') len = 2;
            if ((c1 == U'r' && c2 == U'e') || (c1 == U'v' && c2 == U'e') || (c1 == U'l' && c2 == U'l')) len = 3;
            if (len > 0) {
                emit(i, i + len);
                i += len;
                continue;
            }
        }
        const CharClass cls = classify(c);
        // " ?X+" for letters, numbers and other symbols: one leading ASCII space joins the run.
        if (c == U' ' && i + 1 < n) {
            const CharClass next = classify(cps[i + 1].value);
            if (next != CharClass::Space) {
                const std::size_t end = run_end(i + 1, next);
                emit(i, end);
                i = end;
                continue;
            }
        }
        if (cls != CharClass::Space) {
            const std::size_t end = run_end(i, cls);
            emit(i, end);
            i = end;
            continue;
        }
        // "\s+(?!\S)" then "\s+": a whitespace run before a non-space leaves its last character
        // to the next piece (as " ?X+" when it is a plain space).
        const std::size_t end = run_end(i, CharClass::Space);
        if (end < n && end - i > 1) {
            emit(i, end - 1);
            i = end - 1;
        } else {
            emit(i, end);
            i = end;
        }
    }
    return pieces;
}

std::vector<std::string> ByteLevelBpeTokenizer::bpe(std::string_view piece) const {
    const auto& table = byte_symbols();
    std::vector<std::string> symbols;
    symbols.reserve(piece.size());
    for (char ch : piece) symbols.push_back(table[static_cast<unsigned char>(ch)]);

    // GPT-2: repeatedly merge every occurrence of the lowest-ranked adjacent pair.
    std::string key;
    while (symbols.size() > 1) {
        int best_rank = INT_MAX;
        std::size_t best = 0;
        for (std::size_t k = 0; k + 1 < symbols.size(); ++k) {
            key.assign(symbols[k]).append(1, ' ').append(symbols[k + 1]);
            const auto it = merge_rank_.find(key);
            if (it != merge_rank_.end() && it->second < best_rank) {
                best_rank = it->second;
                best = k;
            }
        }
        if (best_rank == INT_MAX) break;
        const std::string left = symbols[best];
        const std::string right = symbols[best + 1];
        std::vector<std::string> merged;
        merged.reserve(symbols.size());
        for (std::size_t k = 0; k < symbols.size();) {
            if (k + 1 < symbols.size() && symbols[k] == left && symbols[k + 1] == right) {
                merged.push_back(left + right);
                k += 2;
            } else {
                merged.push_back(std::move(symbols[k]));
                ++k;
            }
        }
        symbols = std::move(merged);
    }
    return symbols;
}

std::vector<std::int64_t> ByteLevelBpeTokenizer::encode(std::string_view text, std::size_t max_tokens) const {
    std::vector<std::int64_t> ids{bos_id_};
    const std::size_t limit = max_tokens < 2 ? 0 : max_tokens - 1;  // room for </s>
    for (const std::string& piece : pre_tokenize(text)) {
        for (const std::string& token : bpe(piece)) {
            if (ids.size() >= limit) break;
            const auto it = vocab_.find(token);
            if (it != vocab_.end()) {
                ids.push_back(it->second);
            } else {
                // A complete byte-level vocab has every single-byte symbol; fall back to them.
                for (const CodePoint& cp : decode_utf8(token)) {
                    std::string one;
                    append_utf8(one, cp.value);
                    if (const auto b = vocab_.find(one); b != vocab_.end() && ids.size() < limit) ids.push_back(b->second);
                }
            }
        }
        if (ids.size() >= limit) break;
    }
    ids.push_back(eos_id_);
    return ids;
}

}  // namespace ee
