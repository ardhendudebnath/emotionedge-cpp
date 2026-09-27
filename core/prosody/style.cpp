#include "core/prosody/style.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <random>

#include "core/runtime/params.hpp"

namespace ee {

namespace {

void normalize(StyleVector& v) {
    double norm = 0.0;
    for (float x : v) norm += static_cast<double>(x) * x;
    norm = std::sqrt(norm);
    if (norm <= 1e-12) return;
    for (float& x : v) x = static_cast<float>(x / norm);
}

}  // namespace

StyleBank StyleBank::placeholder(std::uint32_t seed) {
    StyleBank bank;
    std::mt19937 rng(seed);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    for (StyleVector& anchor : bank.anchors_) {
        for (float& x : anchor) x = gauss(rng);
        normalize(anchor);
    }
    return bank;
}

StyleBank StyleBank::load(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) throw ConfigError("cannot open style anchors '" + path.string() + "'");
    nlohmann::json root;
    try {
        in >> root;
    } catch (const nlohmann::json::exception& e) {
        throw ConfigError("style anchors: " + std::string(e.what()));
    }
    StyleBank bank;
    for (std::size_t i = 0; i < kEmotionLabels.size(); ++i) {
        const std::string name(to_string(kEmotionLabels[i]));
        const auto it = root.find(name);
        if (it == root.end() || !it->is_array() || it->size() != kStyleDim) {
            throw ConfigError("style anchors need a " + std::to_string(kStyleDim) + "-float array for '" + name + "'");
        }
        for (std::size_t k = 0; k < kStyleDim; ++k) bank.anchors_[i][k] = (*it)[k].get<float>();
    }
    return bank;
}

const StyleVector& StyleBank::anchor(EmotionLabel label) const {
    return anchors_[static_cast<std::size_t>(label)];
}

StyleVector StyleBank::blend(Vad target, float temperature) const {
    std::array<double, kEmotionLabels.size()> weights{};
    double total = 0.0;
    const double t2 = std::max(1e-3, static_cast<double>(temperature) * temperature);
    for (std::size_t i = 0; i < kEmotionLabels.size(); ++i) {
        const double d = distance(target, prototype(kEmotionLabels[i]));
        weights[i] = std::exp(-d * d / t2);
        total += weights[i];
    }
    StyleVector out{};
    for (std::size_t i = 0; i < kEmotionLabels.size(); ++i) {
        const auto w = static_cast<float>(weights[i] / total);
        for (std::size_t k = 0; k < kStyleDim; ++k) out[k] += w * anchors_[i][k];
    }
    return out;
}

}  // namespace ee
