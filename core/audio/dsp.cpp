#include "core/audio/dsp.hpp"

#include <algorithm>
#include <string>

#include "core/runtime/params.hpp"

namespace ee {

float rms_dbfs(std::span<const float> x) {
    if (x.empty()) return -120.0f;
    double energy = 0.0;
    for (float s : x) energy += static_cast<double>(s) * s;
    const double mean = energy / static_cast<double>(x.size());
    if (mean <= 1e-12) return -120.0f;
    return static_cast<float>(10.0 * std::log10(mean));
}

float peak_abs(std::span<const float> x) {
    float peak = 0.0f;
    for (float s : x) peak = std::max(peak, std::abs(s));
    return peak;
}

float hf_ratio_db(std::span<const float> x) {
    if (x.size() < 2) return 0.0f;
    double energy = 0.0;
    double diff_energy = 0.0;
    for (std::size_t i = 1; i < x.size(); ++i) {
        const double d = static_cast<double>(x[i]) - 0.95 * static_cast<double>(x[i - 1]);
        diff_energy += d * d;
        energy += static_cast<double>(x[i]) * x[i];
    }
    if (energy <= 1e-12) return 0.0f;
    return static_cast<float>(10.0 * std::log10(std::max(diff_energy, 1e-12) / energy));
}

float soft_limit(float x, float ceiling) {
    const float knee = 0.8f * ceiling;
    const float a = std::abs(x);
    if (a <= knee) return x;
    const float span = ceiling - knee;
    const float y = knee + span * std::tanh((a - knee) / span);
    return x < 0.0f ? -y : y;
}

Agc::Agc(Config config, int sample_rate) : cfg_(config), sample_rate_(sample_rate) {}

void Agc::process(std::span<float> block) {
    if (block.empty()) return;
    const float previous_db = gain_db_;
    const float level = rms_dbfs(block);
    if (level > cfg_.gate_dbfs) {
        const float desired = std::clamp(cfg_.target_dbfs - level, cfg_.min_gain_db, cfg_.max_gain_db);
        const float tau = desired < gain_db_ ? cfg_.attack_s : cfg_.release_s;
        const float dt = static_cast<float>(block.size()) / static_cast<float>(sample_rate_);
        const float alpha = 1.0f - std::exp(-dt / std::max(tau, 1e-3f));
        gain_db_ += alpha * (desired - gain_db_);
    }
    // Ramp linearly across the block so gain changes never click.
    const float g0 = db_to_gain(previous_db);
    const float g1 = db_to_gain(gain_db_);
    const float step = (g1 - g0) / static_cast<float>(block.size());
    for (std::size_t i = 0; i < block.size(); ++i) {
        block[i] = soft_limit(block[i] * (g0 + step * static_cast<float>(i + 1)), cfg_.limit);
    }
}

std::unique_ptr<INoiseSuppressor> make_noise_suppressor(std::string_view name) {
    if (name.empty() || name == "none") return nullptr;
    throw ConfigError("noise suppressor '" + std::string(name) +
                      "' is not built in (RNNoise lands with the front-end DSP milestone; use 'none')");
}

std::unique_ptr<IEchoCanceller> make_echo_canceller(std::string_view name) {
    if (name.empty() || name == "none") return nullptr;
    throw ConfigError("echo canceller '" + std::string(name) +
                      "' is not built in (WebRTC AEC3 lands with the front-end DSP milestone; use 'none')");
}

}  // namespace ee
