#include "core/prosody/controller.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

namespace ee {

ControllerConfig ControllerConfig::from(const Params& p) {
    ControllerConfig c;
    c.confidence_floor = p.real("confidence_floor", c.confidence_floor);
    c.confidence_full = p.real("confidence_full", c.confidence_full);
    c.ecs_threshold = p.real("ecs_threshold", c.ecs_threshold);
    c.loop_gain = p.real("loop_gain", c.loop_gain);
    c.loop_decay = p.real("loop_decay", c.loop_decay);
    c.max_correction = p.real("max_correction", c.max_correction);
    if (c.confidence_full <= c.confidence_floor) throw ConfigError("controller: confidence_full must exceed confidence_floor");
    return c;
}

EmotionController::EmotionController(ControllerConfig config) : cfg_(config) {}

float EmotionController::strength(float confidence) const noexcept {
    const float x = std::clamp((confidence - cfg_.confidence_floor) / (cfg_.confidence_full - cfg_.confidence_floor), 0.0f, 1.0f);
    return x * x * (3.0f - 2.0f * x);  // smoothstep
}

Vad EmotionController::target(Vad source) const { return (source + correction_).clamped(); }

ProsodyTargets EmotionController::plan(Vad target, float confidence, const ExpressivityProfile& profile) const {
    // Low confidence shrinks every target toward neutral; the language profile scales intensity.
    const float s = strength(confidence) * profile.intensity;
    const float v = target.v * s;
    const float a = target.a * s;
    const float d = target.d * s;
    const float neg_v = std::max(-v, 0.0f);
    const float pos_a = std::max(a, 0.0f);
    const float pos_d = std::max(d, 0.0f);
    const float neg_d = std::max(-d, 0.0f);

    ProsodyTargets t;
    // Arousal: pitch mean, range, rate and energy all rise with it.
    // Valence below zero lowers the pitch mean, most when arousal is also low (sadness).
    t.pitch_pct = 19.5f * a - 10.0f * neg_v * (1.0f - a) * 0.5f;
    t.range_pct = 38.0f * a * profile.pitch_range;
    t.rate_pct = 13.0f * a * profile.rate;
    t.energy_db = 5.0f * a * profile.energy;
    // Negative valence with high arousal is a tense, pressed voice (anger).
    t.tension = std::clamp(1.4f * neg_v * pos_a, 0.0f, 1.0f);
    // Dominance: falling final contours and fewer hesitation pauses.
    t.final_fall = std::clamp(d, -1.0f, 1.0f);
    t.hesitation = 1.0f - 0.5f * pos_d + 0.4f * neg_d;
    // Emphasis spans: local pitch accent and a 60-100 ms pre-pause, stronger when aroused.
    t.accent = 1.1f + 0.4f * pos_a;
    t.pause_ms = std::clamp(60.0f + 40.0f * pos_a * (1.0f - 0.5f * pos_d), 60.0f, 100.0f);

    // A learned plan for this emotion replaces the rules' rate, pitch, range and contour.
    const EmotionLabel label = nearest_label(target);
    if (const auto it = learned_.controls.find(label); it != learned_.controls.end()) {
        const Vad proto = prototype(label);
        const float proto_sq = proto.v * proto.v + proto.a * proto.a + proto.d * proto.d;
        const float along = proto_sq > 0.0f ? (target.v * proto.v + target.a * proto.a + target.d * proto.d) / proto_sq : 0.0f;
        const float k = std::clamp(along, 0.0f, 1.25f) * strength(confidence) * profile.intensity;
        const LearnedProsody::Controls& c = it->second;
        t.rate_pct = (k * (c.speed - 1.0f)) * 100.0f;
        t.pitch_pct = (std::pow(2.0f, k * c.pitch_st / 12.0f) - 1.0f) * 100.0f;
        t.range_pct = (k * (c.range - 1.0f)) * 100.0f;
        t.final_fall = std::clamp(k * c.fall, -1.0f, 1.0f);
    }
    return t;
}

LearnedProsody LearnedProsody::load(const std::filesystem::path& path, const std::vector<std::string>& only) {
    for (const std::string& name : only) {
        if (!parse_emotion_label(name)) throw ConfigError("learned prosody plan: bad emotion '" + name + "' to keep");
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) throw ConfigError("learned prosody plan not found: " + path.string());
    nlohmann::json root;
    try {
        root = nlohmann::json::parse(in);
    } catch (const nlohmann::json::exception& e) {
        throw ConfigError("learned prosody plan '" + path.string() + "': " + e.what());
    }
    LearnedProsody out;
    const nlohmann::json& controls = root.at("controls");
    for (const auto& name : root.value("adopted", std::vector<std::string>{})) {
        const auto label = parse_emotion_label(name);
        if (!label || !controls.contains(name)) throw ConfigError("learned prosody plan: bad emotion '" + name + "'");
        if (!only.empty() && std::find(only.begin(), only.end(), name) == only.end()) continue;
        const nlohmann::json& c = controls.at(name);
        out.controls[*label] = {c.at("speed").get<float>(), c.at("pitch_st").get<float>(), c.at("range").get<float>(),
                                c.at("fall").get<float>()};
    }
    return out;
}

bool EmotionController::feedback(float ecs, Vad delta, Vad axis_confidence) {
    if (ecs >= cfg_.ecs_threshold) return false;
    // Nudge along -ΔVAD (ΔVAD = VAD_out - VAD_src), weighted by how well each axis was measured.
    const Vad step{-delta.v * axis_confidence.v, -delta.a * axis_confidence.a, -delta.d * axis_confidence.d};
    const Vad next = (correction_ + step * cfg_.loop_gain).clamped(-cfg_.max_correction, cfg_.max_correction);
    const bool changed = !(next == correction_);
    correction_ = next;
    return changed;
}

void EmotionController::next_utterance() { correction_ = correction_ * cfg_.loop_decay; }

}  // namespace ee
