#pragma once

#include <cmath>
#include <cstdint>
#include <numbers>
#include <random>
#include <vector>

namespace ee::test {

inline std::vector<float> silence(double seconds, int rate) {
    return std::vector<float>(static_cast<std::size_t>(seconds * rate), 0.0f);
}

inline std::vector<float> sine(double hz, double seconds, int rate, float amplitude = 0.5f) {
    std::vector<float> out(static_cast<std::size_t>(seconds * rate));
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = amplitude * static_cast<float>(std::sin(2.0 * std::numbers::pi * hz * static_cast<double>(i) / rate));
    }
    return out;
}

/// Harmonic "voiced" buzz with a 1/k spectrum up to 4 kHz.
inline std::vector<float> buzz(double f0, double seconds, int rate, float amplitude = 0.3f) {
    std::vector<float> out(static_cast<std::size_t>(seconds * rate), 0.0f);
    const int harmonics = std::max(1, static_cast<int>(4000.0 / f0));
    for (std::size_t i = 0; i < out.size(); ++i) {
        double v = 0.0;
        for (int k = 1; k <= harmonics; ++k) {
            v += std::sin(2.0 * std::numbers::pi * f0 * k * static_cast<double>(i) / rate) / k;
        }
        out[i] = amplitude * static_cast<float>(v) / 1.8f;
    }
    return out;
}

inline std::vector<float> white_noise(double seconds, int rate, float amplitude, std::uint32_t seed = 7) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-amplitude, amplitude);
    std::vector<float> out(static_cast<std::size_t>(seconds * rate));
    for (float& s : out) s = dist(rng);
    return out;
}

struct SyllableTrain {
    int count = 8;
    double syllable_s = 0.16;
    double gap_s = 0.08;
    double f0_start = 150.0;
    double f0_end = 150.0;
    double rolloff = 1.4;  ///< harmonic k has amplitude 1/k^rolloff: lower = brighter voice
    float amplitude = 0.2f;
    std::vector<float> accent;  ///< optional per-syllable gain in dB
};

/// Speech-like test signal: voiced syllables with raised-cosine envelopes separated by gaps,
/// F0 gliding from f0_start to f0_end over the whole train.
inline std::vector<float> syllable_train(const SyllableTrain& t, int rate) {
    std::vector<float> out;
    const auto syl = static_cast<std::size_t>(t.syllable_s * rate);
    const auto gap = static_cast<std::size_t>(t.gap_s * rate);
    const std::size_t total = static_cast<std::size_t>(t.count) * (syl + gap);
    double phase = 0.0;
    std::size_t n = 0;
    for (int k = 0; k < t.count; ++k) {
        const float gain = static_cast<std::size_t>(k) < t.accent.size() ? std::pow(10.0f, t.accent[k] / 20.0f) : 1.0f;
        for (std::size_t i = 0; i < syl; ++i, ++n) {
            const double f0 = t.f0_start + (t.f0_end - t.f0_start) * static_cast<double>(n) / static_cast<double>(total);
            phase += 2.0 * std::numbers::pi * f0 / rate;
            double v = 0.0;
            for (int h = 1; h * f0 < 4000.0; ++h) v += std::sin(h * phase) / std::pow(h, t.rolloff);
            const double env = std::sin(std::numbers::pi * static_cast<double>(i) / static_cast<double>(syl));
            out.push_back(static_cast<float>(t.amplitude * gain * env * v / 1.6));
        }
        out.insert(out.end(), gap, 0.0f);
        n += gap;
    }
    return out;
}

inline void append(std::vector<float>& dst, const std::vector<float>& src) {
    dst.insert(dst.end(), src.begin(), src.end());
}

inline double rms(const std::vector<float>& x, std::size_t begin = 0, std::size_t end = SIZE_MAX) {
    end = std::min(end, x.size());
    if (end <= begin) return 0.0;
    double e = 0.0;
    for (std::size_t i = begin; i < end; ++i) e += static_cast<double>(x[i]) * x[i];
    return std::sqrt(e / static_cast<double>(end - begin));
}

/// Estimated frequency from positive-going zero crossings.
inline double zero_crossing_hz(const std::vector<float>& x, int rate, std::size_t begin = 0, std::size_t end = SIZE_MAX) {
    end = std::min(end, x.size());
    std::size_t first = 0;
    std::size_t last = 0;
    int crossings = 0;
    for (std::size_t i = begin + 1; i < end; ++i) {
        if (x[i - 1] < 0.0f && x[i] >= 0.0f) {
            if (crossings == 0) first = i;
            last = i;
            ++crossings;
        }
    }
    if (crossings < 2) return 0.0;
    return static_cast<double>(crossings - 1) * rate / static_cast<double>(last - first);
}

}  // namespace ee::test
