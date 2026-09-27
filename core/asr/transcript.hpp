#pragma once

#include <string>
#include <vector>

namespace ee {

/// One recognized word. Times are seconds from the first sample of its utterance's audio,
/// the same origin the emotion engine uses for its energy envelope, so the two can be aligned.
struct Word {
    std::string text;
    float t0 = 0.0f;
    float t1 = 0.0f;
    float probability = 1.0f;
};

/// Joins word texts with single spaces.
inline std::string join_words(const std::vector<Word>& words) {
    std::string out;
    for (const Word& w : words) {
        if (!out.empty()) out += ' ';
        out += w.text;
    }
    return out;
}

}  // namespace ee
