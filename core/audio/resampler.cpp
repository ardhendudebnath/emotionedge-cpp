#include "core/audio/resampler.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <numeric>
#include <stdexcept>

namespace ee {

namespace {

constexpr double kKaiserBeta = 8.6;    // ~-90 dB window sidelobes
constexpr double kRolloff = 0.94;      // pass band ends at 94 % of the lower Nyquist
constexpr std::int64_t kMaxPhases = 1024;

double bessel_i0(double x) {
    double sum = 1.0;
    double term = 1.0;
    const double q = x * x / 4.0;
    for (int k = 1; k < 100; ++k) {
        term *= q / (static_cast<double>(k) * k);
        sum += term;
        if (term < sum * 1e-14) break;
    }
    return sum;
}

double sinc(double x) {
    if (std::abs(x) < 1e-12) return 1.0;
    const double px = std::numbers::pi * x;
    return std::sin(px) / px;
}

}  // namespace

Resampler::Resampler(int in_rate, int out_rate, int zero_crossings) : in_rate_(in_rate), out_rate_(out_rate) {
    if (in_rate <= 0 || out_rate <= 0) throw std::invalid_argument("resampler rates must be positive");
    const std::int64_t g = std::gcd(in_rate, out_rate);
    up_ = out_rate / g;
    down_ = in_rate / g;
    if (passthrough()) return;

    phases_ = std::min(up_, kMaxPhases);
    // Cut-off in cycles per input sample: the lower of the two Nyquist rates, minus a transition band.
    const double cutoff = 0.5 * std::min(1.0, static_cast<double>(up_) / static_cast<double>(down_)) * kRolloff;
    const int zc = std::max(zero_crossings, 2);
    const double reach = zc / (2.0 * cutoff);  // window half-width in input samples
    half_ = static_cast<std::int64_t>(std::ceil(reach));
    const std::int64_t taps = 2 * half_;
    table_.assign(static_cast<std::size_t>(phases_ * taps), 0.0f);

    const double i0_beta = bessel_i0(kKaiserBeta);
    std::vector<double> row(static_cast<std::size_t>(taps));
    for (std::int64_t p = 0; p < phases_; ++p) {
        const double frac = static_cast<double>(p) / static_cast<double>(phases_);
        double sum = 0.0;
        for (std::int64_t k = 0; k < taps; ++k) {
            // Tap k multiplies input sample j = center - half + 1 + k; x is its distance to the output time.
            const double x = frac + static_cast<double>(half_ - 1 - k);
            double h = 0.0;
            if (std::abs(x) < reach) {
                const double r = x / reach;
                const double window = bessel_i0(kKaiserBeta * std::sqrt(1.0 - r * r)) / i0_beta;
                h = 2.0 * cutoff * sinc(2.0 * cutoff * x) * window;
            }
            row[static_cast<std::size_t>(k)] = h;
            sum += h;
        }
        // Unit DC gain for every phase.
        for (std::int64_t k = 0; k < taps; ++k) {
            table_[static_cast<std::size_t>(p * taps + k)] =
                static_cast<float>(sum != 0.0 ? row[static_cast<std::size_t>(k)] / sum : 0.0);
        }
    }
}

void Resampler::reset() {
    buffer_.clear();
    buf_start_ = 0;
    total_in_ = 0;
    next_out_ = 0;
}

void Resampler::produce(std::vector<float>& out, bool flushing) {
    const std::int64_t avail_end = buf_start_ + static_cast<std::int64_t>(buffer_.size());
    const std::int64_t taps = 2 * half_;
    while (true) {
        const std::int64_t t_num = next_out_ * down_;
        if (t_num >= total_in_ * up_) break;  // past the end of the input seen so far
        const std::int64_t center = t_num / up_;
        if (!flushing && center + half_ >= avail_end) break;  // look-ahead not available yet
        const std::int64_t phase = ((t_num % up_) * phases_) / up_;
        const float* row = table_.data() + phase * taps;
        const std::int64_t first = center - half_ + 1;
        const std::int64_t k_lo = std::max<std::int64_t>(0, buf_start_ - first);
        const std::int64_t k_hi = std::min<std::int64_t>(taps, avail_end - first);
        const std::int64_t base = first - buf_start_;  // may be negative; k_lo keeps base + k >= 0
        float acc = 0.0f;
        for (std::int64_t k = k_lo; k < k_hi; ++k) acc += row[k] * buffer_[static_cast<std::size_t>(base + k)];
        out.push_back(acc);
        ++next_out_;
    }
    // Drop input that no future output can reach.
    const std::int64_t keep_from = (next_out_ * down_) / up_ - half_ + 1;
    if (keep_from - buf_start_ > 8192) {
        buffer_.erase(buffer_.begin(), buffer_.begin() + (keep_from - buf_start_));
        buf_start_ = keep_from;
    }
}

void Resampler::process(std::span<const float> in, std::vector<float>& out) {
    if (passthrough()) {
        out.insert(out.end(), in.begin(), in.end());
        return;
    }
    buffer_.insert(buffer_.end(), in.begin(), in.end());
    total_in_ += static_cast<std::int64_t>(in.size());
    produce(out, false);
}

void Resampler::flush(std::vector<float>& out) {
    if (passthrough()) return;
    produce(out, true);
}

std::vector<float> Resampler::convert(std::span<const float> in, int in_rate, int out_rate) {
    Resampler r(in_rate, out_rate);
    std::vector<float> out;
    out.reserve(static_cast<std::size_t>(static_cast<double>(in.size()) * out_rate / in_rate) + 16);
    r.process(in, out);
    r.flush(out);
    return out;
}

}  // namespace ee
