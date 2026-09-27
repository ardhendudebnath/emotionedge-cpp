#include "core/prosody/expressivity.hpp"

#include <yaml-cpp/yaml.h>

#include "core/runtime/params.hpp"

namespace ee {

std::string_view to_string(Register r) noexcept {
    switch (r) {
    case Register::Casual: return "casual";
    case Register::Formal: return "formal";
    case Register::Neutral: return "neutral";
    }
    return "casual";
}

std::optional<Register> parse_register(std::string_view name) noexcept {
    if (name == "casual") return Register::Casual;
    if (name == "formal") return Register::Formal;
    if (name == "neutral") return Register::Neutral;
    return std::nullopt;
}

namespace {

ExpressivityProfile make(std::string lang, float intensity, float range, float rate, Register reg, float ratio) {
    ExpressivityProfile p;
    p.language = std::move(lang);
    p.intensity = intensity;
    p.pitch_range = range;
    p.rate = rate;
    p.reg = reg;
    p.length_ratio = ratio;
    return p;
}

void apply(const YAML::Node& node, ExpressivityProfile& p) {
    if (!node || !node.IsMap()) return;
    try {
        if (node["intensity"]) p.intensity = node["intensity"].as<float>();
        if (node["pitch_range"]) p.pitch_range = node["pitch_range"].as<float>();
        if (node["rate"]) p.rate = node["rate"].as<float>();
        if (node["energy"]) p.energy = node["energy"].as<float>();
        if (node["length_ratio"]) p.length_ratio = node["length_ratio"].as<float>();
        if (node["register"]) {
            const auto r = parse_register(node["register"].as<std::string>());
            if (!r) throw ConfigError("expressivity register must be casual, formal or neutral");
            p.reg = *r;
        }
    } catch (const YAML::Exception& e) {
        throw ConfigError(std::string("expressivity profile: ") + e.what());
    }
}

}  // namespace

ExpressivityProfiles ExpressivityProfiles::defaults() {
    ExpressivityProfiles out;
    out.fallback_ = make("default", 1.0f, 1.0f, 1.0f, Register::Casual, 1.0f);
    for (const ExpressivityProfile& p : {
             make("en", 1.00f, 1.00f, 1.00f, Register::Casual, 1.00f),
             make("hi", 1.00f, 1.00f, 1.00f, Register::Casual, 1.10f),
             make("es", 1.10f, 1.05f, 1.05f, Register::Casual, 1.15f),
             make("it", 1.15f, 1.10f, 1.05f, Register::Casual, 1.10f),
             make("fr", 1.00f, 1.00f, 1.00f, Register::Casual, 1.15f),
             make("de", 0.90f, 0.95f, 1.00f, Register::Casual, 1.00f),
             make("ja", 0.75f, 0.85f, 0.95f, Register::Formal, 1.00f),
             make("ko", 0.80f, 0.85f, 1.00f, Register::Formal, 1.00f),
             make("zh", 0.85f, 0.70f, 1.00f, Register::Casual, 1.00f),  // tonal: keep pitch range tame
             make("vi", 0.85f, 0.70f, 1.00f, Register::Casual, 1.00f),
             make("th", 0.85f, 0.70f, 1.00f, Register::Casual, 1.00f),
         }) {
        out.languages_[p.language] = p;
    }
    return out;
}

ExpressivityProfiles ExpressivityProfiles::load(const std::filesystem::path& path) {
    YAML::Node root;
    try {
        root = YAML::LoadFile(path.string());
    } catch (const YAML::Exception& e) {
        throw ConfigError("cannot read expressivity profiles '" + path.string() + "': " + e.what());
    }
    ExpressivityProfiles out;
    out.fallback_.language = "default";
    apply(root["default"], out.fallback_);
    if (const auto langs = root["languages"]; langs && langs.IsMap()) {
        for (const auto& kv : langs) {
            ExpressivityProfile p = out.fallback_;
            p.language = kv.first.as<std::string>();
            apply(kv.second, p);
            out.languages_[p.language] = p;
        }
    }
    return out;
}

ExpressivityProfile ExpressivityProfiles::get(std::string_view language) const {
    const auto it = languages_.find(language);
    if (it != languages_.end()) return it->second;
    ExpressivityProfile p = fallback_;
    p.language = std::string(language);
    return p;
}

}  // namespace ee
