#include "core/translate/emphasis.hpp"

#include <algorithm>
#include <cmath>

namespace ee {

std::vector<std::uint16_t> project_emphasis(const std::vector<std::uint16_t>& source_emphasis,
                                            std::size_t source_words, std::size_t target_words,
                                            const std::vector<std::uint16_t>& markup,
                                            const std::vector<std::pair<std::uint16_t, std::uint16_t>>& alignment) {
    std::vector<std::uint16_t> out;
    if (source_emphasis.empty() || target_words == 0) return out;
    if (!markup.empty()) return markup;

    for (std::uint16_t s : source_emphasis) {
        bool aligned = false;
        for (const auto& [src, tgt] : alignment) {
            if (src == s && tgt < target_words) {
                out.push_back(tgt);
                aligned = true;
                break;
            }
        }
        if (!aligned && alignment.empty()) {
            const double rel = source_words > 1 ? static_cast<double>(s) / static_cast<double>(source_words - 1) : 0.0;
            out.push_back(static_cast<std::uint16_t>(std::lround(rel * static_cast<double>(target_words - 1))));
        }
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

}  // namespace ee
