#pragma once

#include <cstddef>
#include <deque>
#include <string>
#include <string_view>
#include <vector>

#include "core/asr/transcript.hpp"

namespace ee {

/// LocalAgreement-n streaming policy (blueprint 2.1 "LocalAgreement-2 streaming", after
/// Macháček et al., Whisper-Streaming 2023): a word becomes stable once the last `n`
/// consecutive hypotheses agree on it (as a common prefix after the committed words).
/// Committed words never change, which is what lets captions and wait-k drafts build on them.
class LocalAgreement {
public:
    explicit LocalAgreement(int n = 2);

    /// Feeds the latest full hypothesis for the utterance; returns how many words it committed.
    std::size_t update(const std::vector<Word>& hypothesis);
    /// Final hypothesis at the endpoint: committed words + its (aligned) remainder.
    [[nodiscard]] std::vector<Word> finalize(const std::vector<Word>& hypothesis) const;

    [[nodiscard]] const std::vector<Word>& committed() const noexcept { return committed_; }
    [[nodiscard]] const std::vector<Word>& tentative() const noexcept { return tentative_; }
    /// committed + tentative
    [[nodiscard]] std::vector<Word> current() const;
    void reset();

    /// Lowercase, surrounding punctuation stripped: "This!" == "this".
    [[nodiscard]] static std::string normalize(std::string_view word);

private:
    /// The part of `hypothesis` after the committed prefix.
    [[nodiscard]] std::vector<Word> tail_of(const std::vector<Word>& hypothesis) const;

    int n_;
    std::vector<Word> committed_;
    std::vector<Word> tentative_;
    std::deque<std::vector<Word>> history_;
};

}  // namespace ee
