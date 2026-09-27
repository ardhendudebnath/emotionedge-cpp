#include "core/translate/wait_k.hpp"

#include <algorithm>
#include <cmath>

namespace ee {

WaitKPolicy::WaitKPolicy(int k, float length_ratio) : k_(std::max(1, k)), ratio_(std::max(0.1f, length_ratio)) {}

std::size_t WaitKPolicy::target_budget(std::size_t source_words) const noexcept {
    const auto k = static_cast<std::size_t>(k_);
    if (source_words < k) return 0;
    return static_cast<std::size_t>(std::floor(static_cast<float>(source_words - k + 1) * ratio_));
}

std::size_t WaitKPolicy::extend(std::vector<std::string>& committed, const std::vector<std::string>& draft,
                                std::size_t source_words) const {
    const std::size_t budget = std::min(target_budget(source_words), draft.size());
    std::size_t added = 0;
    for (std::size_t i = committed.size(); i < budget; ++i) {
        committed.push_back(draft[i]);
        ++added;
    }
    return added;
}

}  // namespace ee
