#include "core/pipeline/live_view.hpp"

#include <nlohmann/json.hpp>

namespace ee {

using json = nlohmann::json;

void LiveView::on_event(std::string_view event) {
    const json e = json::parse(event, nullptr, false);
    if (!e.is_object() || !e.contains("utterance") || !e["utterance"].is_number_unsigned()) return;
    const std::string type = e.value("type", "");
    const std::lock_guard<std::mutex> lock(mu_);
    Utterance& u = utterances_[e["utterance"].get<std::uint64_t>()];
    u.id = e["utterance"].get<std::uint64_t>();
    if (type == "transcript") {
        if (u.source_final && !e.value("final", false)) return;  // a late partial after the final
        u.source = e.value("text", "");
        u.source_final = e.value("final", false);
        u.src_start = e.value("start", u.src_start);
        u.src_end = e.value("end", u.src_end);
    } else if (type == "translation") {
        if (u.translation_final && !e.value("final", false)) return;
        u.translation = e.value("text", "");
        u.translation_final = e.value("final", false);
    } else if (type == "emotion") {
        u.emotion = e.value("label", "");
        u.valence = e.value("valence", 0.0f);
        u.arousal = e.value("arousal", 0.0f);
        u.dominance = e.value("dominance", 0.0f);
        u.confidence = e.value("confidence", 0.0f);
        u.emphasis = e.value("emphasis", std::vector<std::string>{});
    } else if (type == "consistency") {
        u.ecs.push_back(e.value("ecs", 0.0f));
        u.heard.push_back(e.contains("heard") ? e["heard"].value("label", "") : "");
    } else if (type == "playout") {
        u.out_start = e.value("start", u.out_start);
        u.out_end = e.value("end", u.out_end);
    } else {
        return;
    }
    version_.fetch_add(1, std::memory_order_release);
}

std::vector<LiveView::Utterance> LiveView::snapshot() const {
    const std::lock_guard<std::mutex> lock(mu_);
    std::vector<Utterance> out;
    out.reserve(utterances_.size());
    for (const auto& [id, u] : utterances_) out.push_back(u);
    return out;
}

void LiveView::clear() {
    const std::lock_guard<std::mutex> lock(mu_);
    utterances_.clear();
    version_.fetch_add(1, std::memory_order_release);
}

}  // namespace ee
