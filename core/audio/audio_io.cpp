#include "core/audio/audio_io.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>

#include "core/runtime/clock.hpp"
#include "core/runtime/thread_util.hpp"

namespace ee {

BufferSource::BufferSource(std::vector<float> samples, int sample_rate)
    : samples_(std::move(samples)), rate_(sample_rate) {}

std::size_t BufferSource::read(std::span<float> out) {
    const std::size_t n = std::min(out.size(), samples_.size() - pos_);
    std::copy_n(samples_.data() + pos_, n, out.data());
    pos_ += n;
    return n;
}

RingSource::RingSource(int sample_rate, std::size_t capacity) : ring_(capacity), rate_(sample_rate) {}

std::size_t RingSource::push(std::span<const float> samples) noexcept {
    const std::size_t n = ring_.write(samples);
    if (n < samples.size()) overflow_.fetch_add(samples.size() - n, std::memory_order_relaxed);
    return n;
}

RingSink::RingSink(int sample_rate, std::size_t capacity) : ring_(capacity), rate_(sample_rate) {}

std::size_t RingSink::pull(std::span<float> out) noexcept {
    if (flush_requested_.exchange(false, std::memory_order_acq_rel)) ring_.skip(ring_.read_available());
    const std::size_t n = ring_.read(out);
    std::fill(out.begin() + static_cast<std::ptrdiff_t>(n), out.end(), 0.0f);
    // Running dry while an utterance is still being delivered is an audible dropout; between
    // utterances an empty ring is just silence.
    if (n < out.size() && streaming_.load(std::memory_order_acquire)) {
        underrun_.fetch_add(out.size() - n, std::memory_order_relaxed);
    }
    // Silence too: the echo canceller's reference must stay continuous to stay aligned.
    if (tap_ != nullptr) tap_->write(out);
    return n;
}

void TimelineSink::write_at(double start_seconds, std::span<const float> samples) {
    const auto start = static_cast<std::size_t>(std::llround(std::max(0.0, start_seconds) * rate_));
    if (start + samples.size() > data_.size()) data_.resize(start + samples.size(), 0.0f);
    for (std::size_t i = 0; i < samples.size(); ++i) data_[start + i] += samples[i];
}

double TimelineSink::end_seconds() const noexcept {
    return static_cast<double>(data_.size()) / static_cast<double>(rate_);
}

PacedFeeder::PacedFeeder(RingSource& target, std::vector<float> samples, double speed, std::size_t block)
    : target_(target), samples_(std::move(samples)), speed_(speed > 0 ? speed : 1.0),
      block_(std::max<std::size_t>(block, 1)) {}

PacedFeeder::~PacedFeeder() { stop(); }

void PacedFeeder::start() {
    if (thread_.joinable()) return;
    thread_ = std::thread([this] {
        set_current_thread_name("T0-feeder");
        set_current_thread_priority(ThreadPriority::Realtime);
        const double block_seconds = static_cast<double>(block_) / target_.sample_rate() / speed_;
        std::vector<float> block(block_);
        TimePoint next = Clock::now();
        for (std::size_t pos = 0; pos < samples_.size() && !stop_.load(); pos += block_) {
            const std::size_t n = std::min(block_, samples_.size() - pos);
            std::copy_n(samples_.data() + pos, n, block.data());
            if (hook_) hook_(std::span<float>(block.data(), n));
            target_.push(std::span<const float>(block.data(), n));
            next = add_seconds(next, block_seconds);
            std::this_thread::sleep_until(next);
        }
        target_.close();
        done_.store(true);
    });
}

void PacedFeeder::stop() {
    stop_.store(true);
    if (thread_.joinable()) thread_.join();
}

PacedDrain::PacedDrain(RingSink& source, double speed, std::size_t block)
    : source_(source), speed_(speed > 0 ? speed : 1.0), block_(std::max<std::size_t>(block, 1)) {}

PacedDrain::~PacedDrain() { stop(); }

void PacedDrain::start() {
    if (thread_.joinable()) return;
    thread_ = std::thread([this] {
        set_current_thread_name("T0-drain");
        set_current_thread_priority(ThreadPriority::Realtime);
        const double block_seconds = static_cast<double>(block_) / source_.sample_rate() / speed_;
        std::vector<float> block(block_);
        TimePoint next = Clock::now();
        while (!stop_.load()) {
            source_.pull(block);
            if (hook_) hook_(block);
            recording_.insert(recording_.end(), block.begin(), block.end());
            next = add_seconds(next, block_seconds);
            std::this_thread::sleep_until(next);
        }
    });
}

void PacedDrain::stop() {
    stop_.store(true);
    if (thread_.joinable()) thread_.join();
}

}  // namespace ee
