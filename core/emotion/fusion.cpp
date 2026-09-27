#include "core/emotion/fusion.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace ee {

FusionConfig FusionConfig::from(const Params& p) {
    FusionConfig cfg;
    cfg.acoustic_weight = {p.real("fusion.acoustic.valence", cfg.acoustic_weight.v),
                           p.real("fusion.acoustic.arousal", cfg.acoustic_weight.a),
                           p.real("fusion.acoustic.dominance", cfg.acoustic_weight.d)};
    cfg.lexical_weight = {p.real("fusion.lexical.valence", cfg.lexical_weight.v),
                          p.real("fusion.lexical.arousal", cfg.lexical_weight.a),
                          p.real("fusion.lexical.dominance", cfg.lexical_weight.d)};
    cfg.gate = p.real("fusion.gate", cfg.gate);
    return cfg;
}

namespace {

struct AxisResult {
    float value = 0.0f;
    float confidence = 0.0f;
};

AxisResult fuse_axis(float a, float ca, float wa, bool a_valid, float l, float cl, float wl, bool l_valid, float gate) {
    const float ga = a_valid && ca >= gate ? 1.0f : 0.0f;
    const float gl = l_valid && cl >= gate ? 1.0f : 0.0f;
    const float wsum_a = ga * wa * ca;
    const float wsum_l = gl * wl * cl;
    AxisResult r;
    if (wsum_a + wsum_l <= 0.0f) return r;
    r.value = (wsum_a * a + wsum_l * l) / (wsum_a + wsum_l);
    r.confidence = 1.0f - (1.0f - ga * ca) * (1.0f - gl * cl);
    return r;
}

}  // namespace

EmotionState fuse(const ModalityEstimate& acoustic, const ModalityEstimate& lexical, const FusionConfig& cfg,
                  Vad* axis_confidence) {
    const auto& a = acoustic;
    const auto& l = lexical;
    const AxisResult v = fuse_axis(a.vad.v, a.confidence.v, cfg.acoustic_weight.v, a.valid, l.vad.v, l.confidence.v,
                                   cfg.lexical_weight.v, l.valid, cfg.gate);
    const AxisResult ar = fuse_axis(a.vad.a, a.confidence.a, cfg.acoustic_weight.a, a.valid, l.vad.a,
                                    l.confidence.a, cfg.lexical_weight.a, l.valid, cfg.gate);
    const AxisResult d = fuse_axis(a.vad.d, a.confidence.d, cfg.acoustic_weight.d, a.valid, l.vad.d, l.confidence.d,
                                   cfg.lexical_weight.d, l.valid, cfg.gate);
    EmotionState out;
    out.vad = Vad{v.value, ar.value, d.value}.clamped();
    out.confidence = (v.confidence + ar.confidence + d.confidence) / 3.0f;
    out.label = nearest_label(out.vad);
    if (axis_confidence != nullptr) *axis_confidence = {v.confidence, ar.confidence, d.confidence};
    return out;
}

EmotionState EmotionSmoother::update(const EmotionState& next, float alpha_override) {
    if (!initialized_) {
        state_ = next;
        initialized_ = true;
        return state_;
    }
    const float base = alpha_override >= 0.0f ? alpha_override : alpha_;
    const float k = std::clamp(base * std::max(next.confidence, 0.1f), 0.0f, 1.0f);
    state_.vad = lerp(state_.vad, next.vad, k);
    state_.confidence += k * (next.confidence - state_.confidence);
    state_.label = nearest_label(state_.vad);
    return state_;
}

EmotionLabel LabelHysteresis::update(Vad point) {
    const EmotionLabel best = nearest_label(point);
    if (!initialized_) {
        initialized_ = true;
        current_ = best;
        return current_;
    }
    if (best == current_) {
        streak_ = 0;
        return current_;
    }
    const float advantage = distance(point, prototype(current_)) - distance(point, prototype(best));
    if (advantage >= strong_margin_) {
        current_ = best;
        streak_ = 0;
    } else if (advantage >= margin_) {
        streak_ = candidate_ == best ? streak_ + 1 : 1;
        candidate_ = best;
        if (streak_ >= updates_) {
            current_ = best;
            streak_ = 0;
        }
    } else {
        streak_ = 0;
    }
    return current_;
}

std::vector<std::uint16_t> find_emphasis(const std::vector<Word>& words, std::span<const float> envelope,
                                         float hop_s, const EmphasisConfig& cfg) {
    std::vector<std::uint16_t> out;
    if (words.size() < 3 || envelope.empty() || hop_s <= 0.0f) return out;

    std::vector<float> peaks(words.size(), -120.0f);
    std::vector<std::size_t> candidates;
    for (std::size_t w = 0; w < words.size(); ++w) {
        const Word& word = words[w];
        if (word.t1 - word.t0 < cfg.min_word_s) continue;
        const auto lo = static_cast<std::size_t>(std::max(0.0f, word.t0 / hop_s));
        const auto hi = std::min(envelope.size(), static_cast<std::size_t>(std::ceil(word.t1 / hop_s)) + 1);
        if (lo >= hi) continue;
        peaks[w] = *std::max_element(envelope.begin() + static_cast<std::ptrdiff_t>(lo),
                                     envelope.begin() + static_cast<std::ptrdiff_t>(hi));
        candidates.push_back(w);
    }
    if (candidates.size() < 3) return out;

    // Remove the utterance's loudness trend first: speakers get quieter toward the end (and a
    // gain control may still be settling), which would otherwise favour the first words.
    double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
    for (std::size_t w : candidates) {
        const double x = 0.5 * (words[w].t0 + words[w].t1);
        sx += x;
        sy += peaks[w];
        sxx += x * x;
        sxy += x * peaks[w];
    }
    const double n = static_cast<double>(candidates.size());
    const double denom = n * sxx - sx * sx;
    const double slope = std::abs(denom) > 1e-12 ? (n * sxy - sx * sy) / denom : 0.0;
    const double intercept = (sy - slope * sx) / n;
    for (std::size_t w : candidates) {
        peaks[w] -= static_cast<float>(intercept + slope * 0.5 * (words[w].t0 + words[w].t1));
    }

    std::vector<float> values;
    for (std::size_t w : candidates) values.push_back(peaks[w]);
    const float mean = std::accumulate(values.begin(), values.end(), 0.0f) / static_cast<float>(values.size());
    float var = 0.0f;
    for (float v : values) var += (v - mean) * (v - mean);
    const float stddev = std::sqrt(var / static_cast<float>(values.size()));
    std::vector<float> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    const float median = sorted[sorted.size() / 2];
    if (stddev < 0.5f) return out;  // flat delivery: nothing stands out

    std::vector<std::pair<float, std::size_t>> scored;
    for (std::size_t w : candidates) {
        const float z = (peaks[w] - mean) / stddev;
        if (z >= cfg.z_threshold && peaks[w] - median >= cfg.min_rise_db) scored.push_back({z, w});
    }
    std::sort(scored.begin(), scored.end(), [](const auto& x, const auto& y) { return x.first > y.first; });
    for (std::size_t i = 0; i < scored.size() && static_cast<int>(i) < cfg.max_words; ++i) {
        out.push_back(static_cast<std::uint16_t>(scored[i].second));
    }
    std::sort(out.begin(), out.end());
    return out;
}

}  // namespace ee
