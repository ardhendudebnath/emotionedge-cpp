#pragma once

#include <atomic>
#include <cstddef>
#include <memory>
#include <utility>

namespace ee {

inline constexpr std::size_t kCacheLine = 64;

[[nodiscard]] constexpr std::size_t next_pow2(std::size_t v) noexcept {
    std::size_t p = 1;
    while (p < v) p <<= 1;
    return p;
}

/// Bounded, lock-free single-producer / single-consumer queue.
///
/// Slots are constructed once and reused in place: the producer fills the slot returned by
/// `acquire()` and calls `publish()`; the consumer reads `front()` and calls `pop()`. Slots are
/// never destroyed while the queue lives, so whatever capacity a slot's members grew (audio
/// buffers, strings) is kept, and steady-state traffic does not allocate. This is how the
/// pipeline carries "pre-allocated tensors" between stages with no locks on the hot path.
template <typename T>
class SpscQueue {
public:
    explicit SpscQueue(std::size_t capacity)
        : capacity_(next_pow2(capacity < 2 ? 2 : capacity)), mask_(capacity_ - 1),
          slots_(std::make_unique<T[]>(capacity_)) {}

    SpscQueue(const SpscQueue&) = delete;
    SpscQueue& operator=(const SpscQueue&) = delete;

    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }

    // ---- producer --------------------------------------------------------------------------

    /// Slot for the next element, or nullptr when full. Fill it, then call publish().
    [[nodiscard]] T* acquire() noexcept {
        const std::size_t tail = tail_.load(std::memory_order_relaxed);
        if (tail - head_cache_ == capacity_) {
            head_cache_ = head_.load(std::memory_order_acquire);
            if (tail - head_cache_ == capacity_) return nullptr;
        }
        return &slots_[tail & mask_];
    }

    /// Makes the slot returned by the last acquire() visible to the consumer.
    void publish() noexcept {
        tail_.store(tail_.load(std::memory_order_relaxed) + 1, std::memory_order_release);
    }

    /// Copy-assigns into the next slot (reusing its capacity). False when full.
    bool try_push(const T& value) {
        T* slot = acquire();
        if (slot == nullptr) return false;
        *slot = value;
        publish();
        return true;
    }

    bool try_push(T&& value) {
        T* slot = acquire();
        if (slot == nullptr) return false;
        *slot = std::move(value);
        publish();
        return true;
    }

    // ---- consumer --------------------------------------------------------------------------

    /// Oldest element, or nullptr when empty. Valid until pop().
    [[nodiscard]] T* front() noexcept { return peek(0); }

    /// Element `i` places behind the front (0 == front), or nullptr if there are not that many.
    [[nodiscard]] T* peek(std::size_t i) noexcept {
        const std::size_t head = head_.load(std::memory_order_relaxed);
        if (tail_cache_ - head <= i) {
            tail_cache_ = tail_.load(std::memory_order_acquire);
            if (tail_cache_ - head <= i) return nullptr;
        }
        return &slots_[(head + i) & mask_];
    }

    /// Releases the front slot back to the producer.
    void pop() noexcept {
        head_.store(head_.load(std::memory_order_relaxed) + 1, std::memory_order_release);
    }

    /// Swaps the front element into `out` so both sides keep their buffers. False when empty.
    bool try_pop(T& out) {
        T* slot = front();
        if (slot == nullptr) return false;
        using std::swap;
        swap(out, *slot);
        pop();
        return true;
    }

    // ---- either side (approximate while the other side runs) --------------------------------

    [[nodiscard]] std::size_t size() const noexcept {
        const std::size_t head = head_.load(std::memory_order_acquire);
        const std::size_t tail = tail_.load(std::memory_order_acquire);
        const std::size_t n = tail - head;
        return n > capacity_ ? capacity_ : n;
    }
    [[nodiscard]] bool empty() const noexcept { return size() == 0; }

    /// Visits every slot; call only before the queue is shared, e.g. to reserve buffer capacity.
    template <typename F>
    void for_each_slot(F&& fn) {
        for (std::size_t i = 0; i < capacity_; ++i) fn(slots_[i]);
    }

private:
    const std::size_t capacity_;
    const std::size_t mask_;
    std::unique_ptr<T[]> slots_;

    // Producer-owned write index and its cached view of the consumer's read index, each on
    // its own cache line so the two threads never false-share.
    alignas(kCacheLine) std::atomic<std::size_t> tail_{0};
    alignas(kCacheLine) std::size_t head_cache_ = 0;
    alignas(kCacheLine) std::atomic<std::size_t> head_{0};
    alignas(kCacheLine) std::size_t tail_cache_ = 0;
};

}  // namespace ee
