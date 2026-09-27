#include "core/audio/pitch.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "core/audio/dsp.hpp"

namespace ee {

Yin::Yin(PitchConfig config) : cfg_(config) {
    tau_min_ = std::max(2, static_cast<int>(std::floor(static_cast<float>(cfg_.sample_rate) / cfg_.fmax_hz)));
    tau_max_ = static_cast<int>(std::ceil(static_cast<float>(cfg_.sample_rate) / cfg_.fmin_hz));
    window_ = cfg_.frame - tau_max_;
    if (window_ < tau_max_ / 2 || tau_min_ >= tau_max_) {
        throw std::invalid_argument("YIN frame too short for the requested pitch range");
    }
    diff_.assign(static_cast<std::size_t>(tau_max_) + 1, 0.0f);
}

PitchFrame Yin::analyze(std::span<const float> x) {
    PitchFrame out;
    if (static_cast<int>(x.size()) < cfg_.frame) return out;
    if (rms_dbfs(x.first(static_cast<std::size_t>(cfg_.frame))) < cfg_.silence_dbfs) return out;

    // Difference function d(tau), then its cumulative-mean normalization d'(tau).
    for (int tau = 1; tau <= tau_max_; ++tau) {
        float d = 0.0f;
        for (int j = 0; j < window_; ++j) {
            const float delta = x[static_cast<std::size_t>(j)] - x[static_cast<std::size_t>(j + tau)];
            d += delta * delta;
        }
        diff_[static_cast<std::size_t>(tau)] = d;
    }
    diff_[0] = 1.0f;
    float running = 0.0f;
    for (int tau = 1; tau <= tau_max_; ++tau) {
        auto& v = diff_[static_cast<std::size_t>(tau)];
        running += v;
        v = running > 0.0f ? v * static_cast<float>(tau) / running : 1.0f;
    }

    // First dip under the absolute threshold, followed down to its local minimum.
    int best = -1;
    for (int tau = tau_min_; tau <= tau_max_; ++tau) {
        if (diff_[static_cast<std::size_t>(tau)] < cfg_.threshold) {
            while (tau + 1 <= tau_max_ && diff_[static_cast<std::size_t>(tau + 1)] < diff_[static_cast<std::size_t>(tau)]) ++tau;
            best = tau;
            break;
        }
    }
    if (best < 0) {
        float min_value = 1.0f;
        for (int tau = tau_min_; tau <= tau_max_; ++tau) min_value = std::min(min_value, diff_[static_cast<std::size_t>(tau)]);
        out.periodicity = 1.0f - min_value;
        return out;  // aperiodic
    }

    float refined = static_cast<float>(best);
    if (best > tau_min_ && best < tau_max_) {
        const float a = diff_[static_cast<std::size_t>(best - 1)];
        const float b = diff_[static_cast<std::size_t>(best)];
        const float c = diff_[static_cast<std::size_t>(best + 1)];
        const float denom = a - 2.0f * b + c;
        if (std::abs(denom) > 1e-12f) refined += 0.5f * (a - c) / denom;
    }
    out.f0_hz = static_cast<float>(cfg_.sample_rate) / refined;
    out.periodicity = 1.0f - diff_[static_cast<std::size_t>(best)];
    out.voiced = out.f0_hz >= cfg_.fmin_hz && out.f0_hz <= cfg_.fmax_hz;
    if (!out.voiced) out.f0_hz = 0.0f;
    return out;
}

std::vector<PitchFrame> track_pitch(std::span<const float> signal, const PitchConfig& config) {
    Yin yin(config);
    std::vector<PitchFrame> frames;
    const auto frame = static_cast<std::size_t>(config.frame);
    const auto hop = static_cast<std::size_t>(config.hop);
    std::vector<float> padded(frame, 0.0f);
    for (std::size_t pos = 0; pos < signal.size(); pos += hop) {
        if (pos + frame <= signal.size()) {
            frames.push_back(yin.analyze(signal.subspan(pos, frame)));
        } else {
            std::fill(padded.begin(), padded.end(), 0.0f);
            std::copy(signal.begin() + static_cast<std::ptrdiff_t>(pos), signal.end(), padded.begin());
            frames.push_back(yin.analyze(padded));
        }
    }
    return frames;
}

float semitones(float f_hz, float ref_hz) {
    if (f_hz <= 0.0f || ref_hz <= 0.0f) return 0.0f;
    return 12.0f * std::log2(f_hz / ref_hz);
}

}  // namespace ee
