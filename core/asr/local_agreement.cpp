#include "core/asr/local_agreement.hpp"

#include <algorithm>
#include <cctype>

namespace ee {

LocalAgreement::LocalAgreement(int n) : n_(std::max(1, n)) {}

void LocalAgreement::reset() {
    committed_.clear();
    tentative_.clear();
    history_.clear();
}

std::string LocalAgreement::normalize(std::string_view word) {
    std::size_t b = 0;
    std::size_t e = word.size();
    const auto is_punct = [](unsigned char c) { return c < 0x80 && std::ispunct(c) != 0 && c != '\''; };
    while (b < e && is_punct(static_cast<unsigned char>(word[b]))) ++b;
    while (e > b && is_punct(static_cast<unsigned char>(word[e - 1]))) --e;
    std::string out(word.substr(b, e - b));
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return static_cast<char>(c < 0x80 ? std::tolower(c) : c); });
    return out;
}

std::vector<Word> LocalAgreement::tail_of(const std::vector<Word>& hyp) const {
    if (committed_.empty()) return hyp;
    // Usual case: the hypothesis repeats the committed words first.
    bool prefix_matches = hyp.size() >= committed_.size();
    for (std::size_t i = 0; prefix_matches && i < committed_.size(); ++i) {
        prefix_matches = normalize(hyp[i].text) == normalize(committed_[i].text);
    }
    if (prefix_matches) return {hyp.begin() + static_cast<std::ptrdiff_t>(committed_.size()), hyp.end()};
    // The engine revised an early word: align on time instead (committed words stay as they are).
    const float cut = committed_.back().t1 - 0.05f;
    std::vector<Word> tail;
    for (const Word& w : hyp) {
        if (w.t0 >= cut) tail.push_back(w);
    }
    return tail;
}

std::size_t LocalAgreement::update(const std::vector<Word>& hypothesis) {
    history_.push_back(tail_of(hypothesis));
    while (history_.size() > static_cast<std::size_t>(n_)) history_.pop_front();

    std::size_t agreed = 0;
    if (history_.size() == static_cast<std::size_t>(n_)) {
        const auto& latest = history_.back();
        for (; agreed < latest.size(); ++agreed) {
            const std::string w = normalize(latest[agreed].text);
            const bool all = std::all_of(history_.begin(), history_.end(), [&](const std::vector<Word>& h) {
                return agreed < h.size() && normalize(h[agreed].text) == w;
            });
            if (!all) break;
        }
        // Commit with the latest timing, then drop the committed words from every tail.
        committed_.insert(committed_.end(), latest.begin(), latest.begin() + static_cast<std::ptrdiff_t>(agreed));
        for (auto& h : history_) h.erase(h.begin(), h.begin() + static_cast<std::ptrdiff_t>(std::min(agreed, h.size())));
    }
    tentative_ = history_.back();
    return agreed;
}

std::vector<Word> LocalAgreement::finalize(const std::vector<Word>& hypothesis) const {
    std::vector<Word> out = committed_;
    const std::vector<Word> tail = tail_of(hypothesis);
    out.insert(out.end(), tail.begin(), tail.end());
    return out;
}

std::vector<Word> LocalAgreement::current() const {
    std::vector<Word> out = committed_;
    out.insert(out.end(), tentative_.begin(), tentative_.end());
    return out;
}

}  // namespace ee
