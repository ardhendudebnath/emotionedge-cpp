#include "core/emotion/lexical.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <unordered_map>
#include <unordered_set>

namespace ee {

namespace {

struct Entry {
    const char* text;
    Vad vad;
};

// Hand-authored V·A·D values in [-1, 1] for common emotional words and phrases. They are a
// placeholder for the trained classifier, not a research lexicon.
constexpr std::array kWords = {
    // anger / hostility
    Entry{"angry", {-0.60f, 0.70f, 0.50f}}, Entry{"furious", {-0.80f, 0.90f, 0.60f}},
    Entry{"mad", {-0.60f, 0.60f, 0.40f}}, Entry{"hate", {-0.80f, 0.60f, 0.40f}},
    Entry{"annoyed", {-0.50f, 0.40f, 0.20f}}, Entry{"outrageous", {-0.60f, 0.70f, 0.40f}},
    Entry{"ridiculous", {-0.50f, 0.50f, 0.30f}}, Entry{"stupid", {-0.60f, 0.40f, 0.30f}},
    Entry{"idiot", {-0.70f, 0.50f, 0.40f}}, Entry{"damn", {-0.40f, 0.60f, 0.30f}},
    Entry{"unacceptable", {-0.60f, 0.50f, 0.50f}}, Entry{"liar", {-0.70f, 0.60f, 0.40f}},
    Entry{"unfair", {-0.60f, 0.50f, 0.10f}}, Entry{"disgusting", {-0.80f, 0.50f, 0.20f}},
    Entry{"awful", {-0.70f, 0.40f, 0.00f}}, Entry{"terrible", {-0.70f, 0.40f, -0.10f}},
    Entry{"worst", {-0.70f, 0.40f, 0.00f}}, Entry{"enough", {-0.30f, 0.50f, 0.60f}},
    Entry{"stop", {-0.30f, 0.50f, 0.50f}}, Entry{"demand", {-0.20f, 0.50f, 0.70f}},
    // fear / anxiety
    Entry{"afraid", {-0.60f, 0.50f, -0.60f}}, Entry{"scared", {-0.60f, 0.60f, -0.60f}},
    Entry{"terrified", {-0.80f, 0.80f, -0.70f}}, Entry{"fear", {-0.60f, 0.50f, -0.50f}},
    Entry{"worried", {-0.50f, 0.40f, -0.40f}}, Entry{"nervous", {-0.40f, 0.50f, -0.40f}},
    Entry{"panic", {-0.70f, 0.80f, -0.60f}}, Entry{"danger", {-0.60f, 0.60f, -0.40f}},
    Entry{"help", {-0.20f, 0.40f, -0.40f}}, Entry{"anxious", {-0.50f, 0.50f, -0.40f}},
    // sadness
    Entry{"sad", {-0.70f, -0.40f, -0.40f}}, Entry{"unhappy", {-0.60f, -0.20f, -0.30f}},
    Entry{"sorry", {-0.40f, -0.20f, -0.30f}}, Entry{"miss", {-0.40f, -0.10f, -0.20f}},
    Entry{"lonely", {-0.60f, -0.40f, -0.50f}}, Entry{"cry", {-0.60f, 0.20f, -0.40f}},
    Entry{"crying", {-0.60f, 0.20f, -0.40f}}, Entry{"tears", {-0.50f, 0.10f, -0.30f}},
    Entry{"lost", {-0.50f, -0.10f, -0.40f}}, Entry{"depressed", {-0.80f, -0.50f, -0.50f}},
    Entry{"tired", {-0.30f, -0.60f, -0.30f}}, Entry{"hopeless", {-0.80f, -0.40f, -0.60f}},
    Entry{"alone", {-0.40f, -0.30f, -0.30f}}, Entry{"died", {-0.80f, 0.00f, -0.40f}},
    Entry{"dead", {-0.70f, 0.00f, -0.30f}}, Entry{"hurt", {-0.60f, 0.20f, -0.30f}},
    Entry{"pain", {-0.60f, 0.30f, -0.30f}}, Entry{"disappointed", {-0.60f, -0.10f, -0.20f}},
    // joy / affection
    Entry{"happy", {0.80f, 0.50f, 0.40f}}, Entry{"glad", {0.70f, 0.30f, 0.30f}},
    Entry{"great", {0.70f, 0.50f, 0.40f}}, Entry{"wonderful", {0.80f, 0.50f, 0.40f}},
    Entry{"amazing", {0.80f, 0.70f, 0.40f}}, Entry{"awesome", {0.80f, 0.70f, 0.40f}},
    Entry{"love", {0.80f, 0.50f, 0.30f}}, Entry{"excited", {0.70f, 0.80f, 0.40f}},
    Entry{"fantastic", {0.80f, 0.70f, 0.40f}}, Entry{"excellent", {0.80f, 0.50f, 0.50f}},
    Entry{"perfect", {0.70f, 0.40f, 0.50f}}, Entry{"thanks", {0.60f, 0.20f, 0.20f}},
    Entry{"yay", {0.80f, 0.80f, 0.40f}}, Entry{"fun", {0.70f, 0.60f, 0.30f}},
    Entry{"beautiful", {0.80f, 0.30f, 0.30f}}, Entry{"delighted", {0.80f, 0.50f, 0.40f}},
    Entry{"congratulations", {0.80f, 0.60f, 0.40f}}, Entry{"won", {0.70f, 0.60f, 0.60f}},
    Entry{"good", {0.50f, 0.20f, 0.30f}}, Entry{"nice", {0.50f, 0.10f, 0.20f}},
    Entry{"best", {0.70f, 0.40f, 0.40f}}, Entry{"proud", {0.70f, 0.40f, 0.60f}},
    Entry{"lucky", {0.60f, 0.40f, 0.20f}},
    // surprise
    Entry{"wow", {0.40f, 0.80f, 0.00f}}, Entry{"whoa", {0.20f, 0.80f, 0.00f}},
    Entry{"surprised", {0.30f, 0.70f, -0.10f}}, Entry{"unbelievable", {0.00f, 0.80f, 0.00f}},
    Entry{"incredible", {0.50f, 0.70f, 0.10f}}, Entry{"shocked", {-0.20f, 0.80f, -0.20f}},
    Entry{"unexpected", {0.10f, 0.60f, -0.10f}},
    // calm / contentment
    Entry{"calm", {0.50f, -0.60f, 0.20f}}, Entry{"relaxed", {0.60f, -0.60f, 0.20f}},
    Entry{"peaceful", {0.60f, -0.60f, 0.20f}}, Entry{"fine", {0.30f, -0.20f, 0.20f}},
    Entry{"okay", {0.20f, -0.20f, 0.10f}}, Entry{"quiet", {0.20f, -0.50f, 0.00f}},
    Entry{"gentle", {0.50f, -0.40f, 0.10f}}, Entry{"comfortable", {0.60f, -0.40f, 0.30f}},
    Entry{"rest", {0.30f, -0.50f, 0.10f}},
};

constexpr std::array kPhrases = {
    Entry{"can't believe", {-0.35f, 0.55f, 0.25f}}, Entry{"cannot believe", {-0.35f, 0.55f, 0.25f}},
    Entry{"how dare", {-0.70f, 0.80f, 0.60f}},      Entry{"no way", {-0.10f, 0.60f, 0.20f}},
    Entry{"thank you", {0.60f, 0.20f, 0.20f}},      Entry{"i'm sorry", {-0.30f, -0.10f, -0.30f}},
    Entry{"oh no", {-0.50f, 0.60f, -0.30f}},        Entry{"shut up", {-0.60f, 0.70f, 0.60f}},
    Entry{"get out", {-0.60f, 0.70f, 0.60f}},       Entry{"well done", {0.70f, 0.40f, 0.40f}},
    Entry{"what the hell", {-0.50f, 0.80f, 0.40f}}, Entry{"leave me alone", {-0.60f, 0.50f, 0.20f}},
    Entry{"i love you", {0.90f, 0.50f, 0.30f}},
};

const std::unordered_set<std::string>& negators() {
    static const std::unordered_set<std::string> set = {
        "not", "no", "never", "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't",
        "can't", "cannot", "won't", "wouldn't", "shouldn't", "hardly", "without"};
    return set;
}

const std::unordered_set<std::string>& intensifiers() {
    static const std::unordered_set<std::string> set = {"very", "so", "really", "extremely", "totally",
                                                        "absolutely", "incredibly", "super", "too"};
    return set;
}

const std::unordered_map<std::string, Vad>& word_table() {
    static const std::unordered_map<std::string, Vad> table = [] {
        std::unordered_map<std::string, Vad> t;
        for (const Entry& e : kWords) t.emplace(e.text, e.vad);
        return t;
    }();
    return table;
}

}  // namespace

std::vector<std::string> tokenize_for_emotion(std::string_view text) {
    std::vector<std::string> tokens;
    std::string current;
    const auto flush = [&] {
        if (!current.empty()) tokens.push_back(std::move(current));
        current.clear();
    };
    for (std::size_t i = 0; i < text.size(); ++i) {
        const auto c = static_cast<unsigned char>(text[i]);
        // U+2019 RIGHT SINGLE QUOTATION MARK (E2 80 99) -> '
        if (c == 0xE2 && i + 2 < text.size() && static_cast<unsigned char>(text[i + 1]) == 0x80 &&
            static_cast<unsigned char>(text[i + 2]) == 0x99) {
            current += '\'';
            i += 2;
        } else if (std::isalnum(c) != 0 || c == '\'' || c >= 0x80) {
            current += static_cast<char>(std::tolower(c));
        } else if (c == '!' || c == '?') {
            flush();
            tokens.emplace_back(1, static_cast<char>(c));
        } else {
            flush();
        }
    }
    flush();
    return tokens;
}

bool LexiconEmotionModel::supports(std::string_view language) const {
    return language.empty() || language == "en";
}

ModalityEstimate LexiconEmotionModel::estimate(std::string_view text, std::string_view language) {
    ModalityEstimate e;
    if (!supports(language)) return e;
    const std::vector<std::string> tokens = tokenize_for_emotion(text);

    int exclamations = 0;
    int questions = 0;
    for (const std::string& t : tokens) {
        exclamations += t == "!" ? 1 : 0;
        questions += t == "?" ? 1 : 0;
    }
    // Shouted words: two or more letters, all upper case, in the original text.
    int shouted = 0;
    {
        int letters = 0;
        bool all_upper = true;
        for (std::size_t i = 0; i <= text.size(); ++i) {
            const auto c = i < text.size() ? static_cast<unsigned char>(text[i]) : ' ';
            if (std::isalpha(c) != 0) {
                ++letters;
                all_upper = all_upper && std::isupper(c) != 0;
            } else if (c != '\'') {
                if (letters >= 2 && all_upper) ++shouted;
                letters = 0;
                all_upper = true;
            }
        }
    }

    Vad sum;
    int hits = 0;
    const auto& table = word_table();
    for (std::size_t i = 0; i < tokens.size();) {
        std::size_t matched_len = 0;
        Vad value;
        for (const Entry& phrase : kPhrases) {  // longest-first is unnecessary: phrases do not overlap
            const std::string_view p = phrase.text;
            std::string joined;
            std::size_t len = 0;
            for (std::size_t k = i; k < tokens.size() && joined.size() < p.size(); ++k, ++len) {
                if (!joined.empty()) joined += ' ';
                joined += tokens[k];
            }
            if (joined == p) {
                matched_len = len;
                value = phrase.vad;
                break;
            }
        }
        if (matched_len == 0) {
            if (const auto it = table.find(tokens[i]); it != table.end()) {
                matched_len = 1;
                value = it->second;
            }
        }
        if (matched_len == 0) {
            ++i;
            continue;
        }
        if (i > 0 && intensifiers().contains(tokens[i - 1])) value = value * 1.4f;
        bool negated = false;
        for (std::size_t k = i >= 3 ? i - 3 : 0; k < i; ++k) negated = negated || negators().contains(tokens[k]);
        if (negated) value = {-0.6f * value.v, 0.7f * value.a, 0.8f * value.d};
        sum = sum + value.clamped();
        ++hits;
        i += matched_len;
    }

    const float punctuation_arousal =
        0.12f * static_cast<float>(std::min(exclamations, 3)) + 0.15f * static_cast<float>(std::min(shouted, 2)) +
        (exclamations > 0 && questions > 0 ? 0.1f : 0.0f);
    if (hits == 0 && punctuation_arousal == 0.0f) return e;  // nothing to say: abstain

    const Vad mean = hits > 0 ? sum * (1.0f / static_cast<float>(hits)) : Vad{};
    e.vad = Vad{mean.v, mean.a + punctuation_arousal, mean.d}.clamped();
    const float word_conf = 0.85f * (1.0f - std::exp(-static_cast<float>(hits) / 1.2f));
    e.confidence = {word_conf, std::max(word_conf, punctuation_arousal > 0.0f ? 0.25f : 0.0f), 0.8f * word_conf};
    e.valid = true;
    return e;
}

std::unique_ptr<ILexicalEmotionModel> make_lexical_model(const Params& params, const ModelRegistry* registry) {
    (void)registry;
    const std::string kind = params.str("lexical", "lexicon");
    if (kind == "lexicon") return std::make_unique<LexiconEmotionModel>();
    if (kind == "none") return nullptr;
    throw ConfigError("unknown lexical emotion model '" + kind +
                      "' (lexicon | none; DistilRoBERTa arrives with roadmap phase 2)");
}

}  // namespace ee
