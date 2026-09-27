#include "core/emotion/class_mapping.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>

#include "core/runtime/params.hpp"

namespace ee {

ClassEmotionMap ClassEmotionMap::from_file(const std::filesystem::path& labels_json) {
    std::ifstream in(labels_json, std::ios::binary);
    if (!in) throw ConfigError("cannot open emotion label map '" + labels_json.string() + "'");
    const std::string where = "emotion label map '" + labels_json.string() + "'";
    nlohmann::json j;
    try {
        in >> j;
    } catch (const nlohmann::json::exception& e) {
        throw ConfigError(where + ": " + e.what());
    }
    ClassEmotionMap map;
    try {
        map.labels = j.at("labels").get<std::vector<std::string>>();
        std::vector<std::string> abstain_names = j.value("abstain", std::vector<std::string>{});
        const std::string neutral_name = j.value("neutral", std::string("neutral"));
        if (j.contains("reliability")) {
            const auto r = j.at("reliability").get<std::vector<float>>();
            if (r.size() != 3) throw ConfigError(where + ": 'reliability' needs 3 values");
            map.reliability = Vad{r[0], r[1], r[2]}.clamped(0.0f, 1.0f);
        }
        const auto& positions = j.at("vad");
        for (std::size_t k = 0; k < map.labels.size(); ++k) {
            const std::string& name = map.labels[k];
            const bool skip = std::find(abstain_names.begin(), abstain_names.end(), name) != abstain_names.end();
            map.abstain.push_back(skip);
            if (name == neutral_name) map.neutral = k;
            if (skip) {
                map.vad.push_back({});
                continue;
            }
            if (!positions.contains(name)) throw ConfigError(where + ": no V·A·D position for class '" + name + "'");
            const auto p = positions.at(name).get<std::vector<float>>();
            if (p.size() != 3) throw ConfigError(where + ": class '" + name + "' needs [v, a, d]");
            map.vad.push_back(Vad{p[0], p[1], p[2]}.clamped());
        }
    } catch (const nlohmann::json::exception& e) {
        throw ConfigError(where + ": " + e.what());
    }
    if (map.labels.empty()) throw ConfigError(where + ": no labels");
    return map;
}

ModalityEstimate ClassEmotionMap::from_probabilities(std::span<const float> probabilities) const {
    ModalityEstimate e;
    if (probabilities.size() != labels.size()) return e;
    double evidence = 0.0;  // mass outside the abstain classes
    std::size_t classes = 0;
    for (std::size_t k = 0; k < labels.size(); ++k) {
        if (abstain[k]) continue;
        evidence += std::max(0.0f, probabilities[k]);
        ++classes;
    }
    if (evidence < 1e-6 || classes == 0) return e;

    Vad point;
    double entropy = 0.0;
    for (std::size_t k = 0; k < labels.size(); ++k) {
        if (abstain[k]) continue;
        const double p = std::max(0.0f, probabilities[k]) / evidence;
        point = point + vad[k] * static_cast<float>(p);
        if (p > 0.0) entropy -= p * std::log(p);
    }
    const double certainty = classes > 1 ? 1.0 - entropy / std::log(static_cast<double>(classes)) : 1.0;
    const double neutral_share = neutral != kNone && !abstain[neutral]
                                     ? std::max(0.0f, probabilities[neutral]) / evidence
                                     : 0.0;
    const double strength = 1.0 - neutral_share;
    const auto scale = static_cast<float>((0.3 + 0.7 * strength) * (0.5 + 0.5 * certainty) * std::min(1.0, evidence));
    e.vad = point.clamped();
    e.confidence = Vad{reliability.v * scale, reliability.a * scale, reliability.d * scale}.clamped(0.0f, 1.0f);
    e.valid = true;
    return e;
}

ModalityEstimate ClassEmotionMap::from_logits(std::span<const float> logits) const {
    if (logits.empty()) return {};
    const float top = *std::max_element(logits.begin(), logits.end());
    std::vector<float> p(logits.size());
    double sum = 0.0;
    for (std::size_t k = 0; k < logits.size(); ++k) {
        p[k] = std::exp(logits[k] - top);
        sum += p[k];
    }
    for (float& x : p) x = static_cast<float>(x / sum);
    return from_probabilities(p);
}

}  // namespace ee
