#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace ee {

/// Wait-k simultaneous-translation policy (blueprint 3.2 "wait-k drafts on partials"): after
/// reading s stable source words, at most (s - k + 1) · length_ratio target words may be
/// committed. The final pass re-translates the whole sentence, keeping that prefix.
class WaitKPolicy {
public:
    explicit WaitKPolicy(int k = 3, float length_ratio = 1.0f);

    /// Target words that may be committed after `source_words` stable source words.
    [[nodiscard]] std::size_t target_budget(std::size_t source_words) const noexcept;

    /// Grows `committed` from a new draft: the committed prefix is kept even if the draft
    /// disagrees with it, and new words are appended up to the budget. Returns words added.
    std::size_t extend(std::vector<std::string>& committed, const std::vector<std::string>& draft,
                       std::size_t source_words) const;

    [[nodiscard]] int k() const noexcept { return k_; }

private:
    int k_;
    float ratio_;
};

}  // namespace ee
