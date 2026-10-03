#include "core/audio/echo_sim.hpp"

#include <algorithm>
#include <cmath>
#include <random>

#include "core/audio/dsp.hpp"

namespace ee {

namespace {

// Direct path, three early reflections, then a decaying noise tail (-60 dB at rt60).
std::vector<float> room_response(int rate, float rt60_ms, std::uint32_t seed) {
    const auto length = std::max<std::size_t>(1, static_cast<std::size_t>(rate * rt60_ms / 1000.0f));
    std::vector<float> h(length, 0.0f);
    h[0] = 1.0f;
    const std::pair<float, float> reflections[] = {{3.0f, 0.5f}, {7.0f, -0.35f}, {11.0f, 0.25f}};
    for (const auto& [ms, amplitude] : reflections) {
        const auto k = static_cast<std::size_t>(rate * ms / 1000.0f);
        if (k < length) h[k] += amplitude;
    }
    std::mt19937 rng(seed);
    std::normal_distribution<float> noise(0.0f, 1.0f);
    const auto start = static_cast<std::size_t>(rate * 0.005f);
    const float decay = 6.9078f / (rt60_ms / 1000.0f);  // ln(1000): -60 dB at rt60
    for (std::size_t k = start; k < length; ++k) {
        h[k] += 0.15f * noise(rng) * std::exp(-decay * static_cast<float>(k) / static_cast<float>(rate));
    }
    return h;
}

}  // namespace

EchoSimulator::EchoSimulator(Config config, int playback_rate, int capture_rate)
    : cfg_(config), gain_(db_to_gain(config.gain_db)), resampler_(playback_rate, capture_rate),
      ring_(static_cast<std::size_t>(capture_rate) * 4),
      rir_(room_response(capture_rate, config.rt60_ms, config.seed)), history_(rir_.size() - 1, 0.0f) {
    // The ring's prefill is the delay: both ends run at the same real-time pace.
    const std::vector<float> delay(static_cast<std::size_t>(capture_rate * config.delay_ms / 1000.0f), 0.0f);
    ring_.write(delay);
}

void EchoSimulator::played(std::span<const float> samples) {
    resampled_.clear();
    resampler_.process(samples, resampled_);
    if (cfg_.drive > 0.0f) {
        for (float& x : resampled_) x = std::tanh(cfg_.drive * x) / cfg_.drive;
    }
    ring_.write(resampled_);  // a full ring drops echo, never blocks the playback thread
}

void EchoSimulator::add_to(std::span<float> capture) {
    block_.resize(capture.size());
    const std::size_t got = ring_.read(block_);
    std::fill(block_.begin() + static_cast<std::ptrdiff_t>(got), block_.end(), 0.0f);
    // y[n] = sum_k h[k] x[n - k], over the history and this block.
    const std::size_t taps = rir_.size();
    const std::size_t past = history_.size();
    for (std::size_t n = 0; n < block_.size(); ++n) {
        float y = 0.0f;
        for (std::size_t k = 0; k < taps; ++k) {
            const std::ptrdiff_t i = static_cast<std::ptrdiff_t>(n) - static_cast<std::ptrdiff_t>(k);
            const float x = i >= 0 ? block_[static_cast<std::size_t>(i)]
                                   : history_[static_cast<std::size_t>(static_cast<std::ptrdiff_t>(past) + i)];
            y += rir_[k] * x;
        }
        capture[n] += gain_ * y;
    }
    // Keep the newest `past` inputs.
    if (block_.size() >= past) {
        std::copy(block_.end() - static_cast<std::ptrdiff_t>(past), block_.end(), history_.begin());
    } else {
        std::rotate(history_.begin(), history_.begin() + static_cast<std::ptrdiff_t>(block_.size()), history_.end());
        std::copy(block_.begin(), block_.end(), history_.end() - static_cast<std::ptrdiff_t>(block_.size()));
    }
}

}  // namespace ee
