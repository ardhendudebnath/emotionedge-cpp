#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace ee {

/// Catch-up playout (5.1): plays queued speech faster without changing its pitch, by WSOLA
/// (waveform-similarity overlap-add). Each output hop overlap-adds the input segment, near where
/// the speed puts it, that best continues the waveform already played. Real-time safe once
/// built: the playback callback (T0) runs it, and it never allocates or locks.
///
/// At speed 1 it passes samples through untouched. It enters and leaves compression seamlessly
/// (the first and last overlaps fade between the plain input and the chosen segment), ramps the
/// speed rather than jumping, and never compresses input at or after `limit`: there the next
/// utterance starts, and its first audio plays as synthesized, on time.
///
/// Positions are absolute: input samples since construction, counting dropped ones (flush),
/// so they line up with the samples written to the queue.
class TimeCompressor {
public:
    static constexpr std::uint64_t kNoLimit = std::numeric_limits<std::uint64_t>::max();

    /// `window_ms` is the overlap-add frame (two hops); `seek_ms` how far either side of its
    /// nominal place a segment may be taken from. `ramp_per_s` caps how fast the speed changes.
    explicit TimeCompressor(int sample_rate, double window_ms = 30.0, double seek_ms = 6.0, double ramp_per_s = 1.5);

    /// Where up to `want` more input samples go; commit() how many were written there.
    [[nodiscard]] std::span<float> input_space(std::size_t want) noexcept;
    void commit(std::size_t n) noexcept;
    /// Input staged and not yet played.
    [[nodiscard]] std::size_t staged() const noexcept { return end_ - begin_; }
    /// How much input to keep staged to fill `out` samples at full speed.
    [[nodiscard]] std::size_t lookahead(std::size_t out) const noexcept;

    /// Fills `out` from the staged input at up to `speed` (>= 1) times real time and returns
    /// how many samples it wrote: fewer once the staged input runs out.
    std::size_t produce(std::span<float> out, double speed, std::uint64_t limit = kNoLimit) noexcept;

    /// True when nothing is staged or held: the caller may bypass produce() (and advance()).
    [[nodiscard]] bool idle() const noexcept { return !compressing_ && staged() == 0 && carry_pos_ == carry_len_; }
    /// Input that bypassed produce() while idle, played as is.
    void advance(std::size_t n) noexcept { base_ += n; }
    /// Drops everything staged plus `skipped` samples taken off the queue unplayed (barge-in).
    void flush(std::size_t skipped) noexcept;

    /// Absolute index of the next input sample to play: everything before it has played or been
    /// skipped over.
    [[nodiscard]] std::uint64_t position() const noexcept { return base_ + begin_; }
    [[nodiscard]] bool compressing() const noexcept { return compressing_; }
    /// The speed of the last hop (1 when passing through).
    [[nodiscard]] double speed() const noexcept { return compressing_ ? speed_ : 1.0; }

private:
    /// Whether a frame at the next nominal place fits the staged input and the limit.
    [[nodiscard]] bool frame_fits(double nominal, std::uint64_t limit) const noexcept;
    void enter() noexcept;
    void frame() noexcept;
    [[nodiscard]] std::size_t best_offset(std::size_t natural, std::size_t lo, std::size_t hi) const noexcept;

    std::size_t hop_;       // H: output samples per frame; the window is 2H
    std::size_t seek_;      // how far either side of the nominal place a segment may start
    double ramp_per_hop_;   // the largest speed change per hop
    std::vector<float> window_;  // sin^2 over 2H: w[n] + w[n + H] = 1

    std::vector<float> buf_;     // staged input: [begin_, end_) is unplayed, buf_[0] is input base_
    std::size_t begin_ = 0;
    std::size_t end_ = 0;
    std::uint64_t base_ = 0;

    bool compressing_ = false;
    double speed_ = 1.0;
    double nominal_ = 0.0;       // where the next frame would start at exactly `speed_`, absolute
    std::vector<float> tail_;    // the previous frame's second half, windowed: the next hop's fade-out
    std::vector<float> carry_;   // one hop of output not yet handed out
    std::size_t carry_pos_ = 0;
    std::size_t carry_len_ = 0;
};

}  // namespace ee
