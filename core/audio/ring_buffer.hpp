#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <memory>
#include <span>

#include "core/runtime/spsc_queue.hpp"

namespace ee {

/// Lock-free single-producer / single-consumer ring of samples (blueprint 1.1: "lock-free SPSC
/// ring buffer"). The audio callback writes without blocking or allocating; the DSP thread reads.
template <typename T>
class SpscRing {
public:
    explicit SpscRing(std::size_t capacity)
        : capacity_(next_pow2(capacity < 2 ? 2 : capacity)), mask_(capacity_ - 1),
          data_(std::make_unique<T[]>(capacity_)) {}

    SpscRing(const SpscRing&) = delete;
    SpscRing& operator=(const SpscRing&) = delete;

    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }

    /// Producer: copies as many samples as fit; returns the number written.
    std::size_t write(std::span<const T> src) noexcept {
        const std::size_t tail = tail_.load(std::memory_order_relaxed);
        const std::size_t head = head_.load(std::memory_order_acquire);
        const std::size_t n = std::min(src.size(), capacity_ - (tail - head));
        const std::size_t idx = tail & mask_;
        const std::size_t first = std::min(n, capacity_ - idx);
        std::copy_n(src.data(), first, data_.get() + idx);
        std::copy_n(src.data() + first, n - first, data_.get());
        tail_.store(tail + n, std::memory_order_release);
        return n;
    }

    /// Consumer: copies up to dst.size() samples; returns the number read.
    std::size_t read(std::span<T> dst) noexcept {
        const std::size_t head = head_.load(std::memory_order_relaxed);
        const std::size_t tail = tail_.load(std::memory_order_acquire);
        const std::size_t n = std::min(dst.size(), tail - head);
        const std::size_t idx = head & mask_;
        const std::size_t first = std::min(n, capacity_ - idx);
        std::copy_n(data_.get() + idx, first, dst.data());
        std::copy_n(data_.get(), n - first, dst.data() + first);
        head_.store(head + n, std::memory_order_release);
        return n;
    }

    /// Consumer: discards up to n samples; returns the number discarded.
    std::size_t skip(std::size_t n) noexcept {
        const std::size_t head = head_.load(std::memory_order_relaxed);
        const std::size_t tail = tail_.load(std::memory_order_acquire);
        n = std::min(n, tail - head);
        head_.store(head + n, std::memory_order_release);
        return n;
    }

    [[nodiscard]] std::size_t read_available() const noexcept {
        return tail_.load(std::memory_order_acquire) - head_.load(std::memory_order_acquire);
    }
    [[nodiscard]] std::size_t write_available() const noexcept { return capacity_ - read_available(); }

private:
    const std::size_t capacity_;
    const std::size_t mask_;
    std::unique_ptr<T[]> data_;
    alignas(kCacheLine) std::atomic<std::size_t> tail_{0};
    alignas(kCacheLine) std::atomic<std::size_t> head_{0};
};

}  // namespace ee
