#include "core/audio/time_compressor.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace ee {

TimeCompressor::TimeCompressor(int sample_rate, double window_ms, double seek_ms, double ramp_per_s)
    : hop_(static_cast<std::size_t>(std::max(1L, std::lround(sample_rate * window_ms / 2000.0)))),
      seek_(static_cast<std::size_t>(std::max(0L, std::lround(sample_rate * seek_ms / 1000.0)))),
      ramp_per_hop_(ramp_per_s * static_cast<double>(hop_) / sample_rate),
      window_(2 * hop_),
      buf_(std::max<std::size_t>(static_cast<std::size_t>(sample_rate), 16 * (2 * hop_ + seek_))),
      tail_(hop_),
      carry_(hop_) {
    for (std::size_t n = 0; n < window_.size(); ++n) {
        const double s = std::sin(std::numbers::pi * static_cast<double>(n) / static_cast<double>(window_.size()));
        window_[n] = static_cast<float>(s * s);
    }
}

std::size_t TimeCompressor::lookahead(std::size_t out) const noexcept {
    // Two output blocks at up to 2x, plus a frame either side of the seek range.
    return std::min(buf_.size(), 2 * out + 4 * hop_ + 2 * seek_);
}

std::span<float> TimeCompressor::input_space(std::size_t want) noexcept {
    if (buf_.size() - end_ < want && begin_ > 0) {
        std::copy(buf_.begin() + static_cast<std::ptrdiff_t>(begin_), buf_.begin() + static_cast<std::ptrdiff_t>(end_),
                  buf_.begin());
        base_ += begin_;
        end_ -= begin_;
        begin_ = 0;
    }
    return {buf_.data() + end_, std::min(want, buf_.size() - end_)};
}

void TimeCompressor::commit(std::size_t n) noexcept { end_ = std::min(buf_.size(), end_ + n); }

void TimeCompressor::flush(std::size_t skipped) noexcept {
    base_ += end_ + skipped;
    begin_ = end_ = 0;
    compressing_ = false;
    speed_ = 1.0;
    carry_pos_ = carry_len_ = 0;
}

bool TimeCompressor::frame_fits(double nominal, std::uint64_t limit) const noexcept {
    // The furthest segment the next frame may take ends 2H after nominal + seek.
    const std::uint64_t end = static_cast<std::uint64_t>(std::max(0.0, nominal)) + seek_ + 2 * hop_;
    return end <= base_ + end_ && end <= limit;
}

std::size_t TimeCompressor::produce(std::span<float> out, double speed, std::uint64_t limit) noexcept {
    speed = std::clamp(speed, 1.0, 2.0);
    std::size_t written = 0;
    while (written < out.size()) {
        if (carry_pos_ < carry_len_) {
            const std::size_t n = std::min(out.size() - written, carry_len_ - carry_pos_);
            std::copy_n(carry_.data() + carry_pos_, n, out.data() + written);
            carry_pos_ += n;
            written += n;
            continue;
        }
        if (!compressing_) {
            const double first = std::min(speed, 1.0 + ramp_per_hop_);
            if (speed > 1.0 && frame_fits(static_cast<double>(position()) + (first - 1.0) * static_cast<double>(hop_), limit)) {
                enter();
                continue;
            }
            const std::size_t n = std::min(out.size() - written, staged());
            if (n == 0) break;
            std::copy_n(buf_.data() + begin_, n, out.data() + written);
            begin_ += n;
            written += n;
            continue;
        }
        const double next = speed_ < speed ? std::min(speed, speed_ + ramp_per_hop_) : std::max(speed, speed_ - ramp_per_hop_);
        if (next <= 1.0 + 1e-9 || !frame_fits(nominal_ + next * static_cast<double>(hop_), limit)) {
            // The plain input from position() on is exactly what the next frame would add to the
            // held tail at speed 1 (w[n] + w[n + H] = 1), so stopping here is seamless.
            compressing_ = false;
            speed_ = 1.0;
            continue;
        }
        speed_ = next;
        frame();
    }
    return written;
}

void TimeCompressor::enter() noexcept {
    // As if a frame had started H before position(): its second half is the plain input from
    // here, fading out under the first real frame.
    for (std::size_t m = 0; m < hop_; ++m) tail_[m] = buf_[begin_ + m] * window_[hop_ + m];
    nominal_ = static_cast<double>(position()) - static_cast<double>(hop_);
    speed_ = 1.0;
    compressing_ = true;
}

void TimeCompressor::frame() noexcept {
    nominal_ += speed_ * static_cast<double>(hop_);
    // Only forward of the natural continuation (no input is played twice), within seek of nominal.
    const std::uint64_t natural = position();
    const auto centre = static_cast<std::uint64_t>(nominal_);
    const std::uint64_t lo = std::max(natural, centre > seek_ ? centre - seek_ : 0);
    const std::uint64_t hi = std::max(lo, centre + seek_);
    const std::size_t a = best_offset(begin_, static_cast<std::size_t>(lo - base_), static_cast<std::size_t>(hi - base_));
    for (std::size_t m = 0; m < hop_; ++m) {
        carry_[m] = tail_[m] + buf_[a + m] * window_[m];
        tail_[m] = buf_[a + hop_ + m] * window_[hop_ + m];
    }
    begin_ = a + hop_;
    carry_pos_ = 0;
    carry_len_ = hop_;
}

std::size_t TimeCompressor::best_offset(std::size_t natural, std::size_t lo, std::size_t hi) const noexcept {
    // The segment most like the plain continuation, by normalized cross-correlation over a hop
    // (the stretch the two overlap).
    const float* ref = buf_.data() + natural;
    const float* x = buf_.data();
    double energy = 0.0;
    for (std::size_t m = 0; m < hop_; ++m) energy += static_cast<double>(x[lo + m]) * x[lo + m];
    std::size_t best = lo;
    double best_score = -1e300;
    for (std::size_t a = lo; a <= hi; ++a) {
        if (a > lo) {
            energy += static_cast<double>(x[a + hop_ - 1]) * x[a + hop_ - 1] - static_cast<double>(x[a - 1]) * x[a - 1];
        }
        float dot = 0.0f;
        for (std::size_t m = 0; m < hop_; ++m) dot += ref[m] * x[a + m];
        const double score = dot / std::sqrt(std::max(energy, 1e-12));
        if (score > best_score) {
            best_score = score;
            best = a;
        }
    }
    return best;
}

}  // namespace ee
