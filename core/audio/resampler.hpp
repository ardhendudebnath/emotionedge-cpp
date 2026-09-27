#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ee {

/// Streaming band-limited resampler: a Kaiser-windowed sinc evaluated from a precomputed
/// polyphase table (exact for rational ratios with up to 1024 phases, finely quantized above).
/// Used for device/file rates -> 16 kHz and for TTS output (22.05/24 kHz) -> playback rate.
class Resampler {
public:
    /// `zero_crossings` per side trades CPU for stopband attenuation (16 ≈ 80+ dB).
    Resampler(int in_rate, int out_rate, int zero_crossings = 16);

    [[nodiscard]] int in_rate() const noexcept { return in_rate_; }
    [[nodiscard]] int out_rate() const noexcept { return out_rate_; }
    [[nodiscard]] bool passthrough() const noexcept { return in_rate_ == out_rate_; }

    /// Appends the output for `in` to `out`. State carries across calls, so a stream can be
    /// fed in any chunk sizes with identical results.
    void process(std::span<const float> in, std::vector<float>& out);
    /// Emits the remaining output at end of stream (the filter's look-ahead is zero-padded).
    void flush(std::vector<float>& out);
    void reset();

    /// One-shot conversion of a whole signal.
    [[nodiscard]] static std::vector<float> convert(std::span<const float> in, int in_rate, int out_rate);

private:
    void produce(std::vector<float>& out, bool flushing);

    int in_rate_;
    int out_rate_;
    std::int64_t up_ = 1;    // L: output = input * L / M
    std::int64_t down_ = 1;  // M
    std::int64_t phases_ = 1;
    std::int64_t half_ = 1;  // taps per side, in input samples
    std::vector<float> table_;  // phases_ x (2 * half_)

    std::vector<float> buffer_;      // input samples from absolute index buf_start_
    std::int64_t buf_start_ = 0;
    std::int64_t total_in_ = 0;
    std::int64_t next_out_ = 0;
};

}  // namespace ee
