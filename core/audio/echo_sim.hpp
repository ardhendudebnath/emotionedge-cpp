#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "core/audio/resampler.hpp"
#include "core/audio/ring_buffer.hpp"

namespace ee {

/// A loudspeaker -> room -> microphone path for real-time runs without hardware. What the
/// speaker plays comes back into the capture, delayed, soft-clipped by a small loudspeaker and
/// reverberated, the way an open loudspeaker feeds a microphone (the problem 1.2's echo
/// canceller solves). Runs the whole loop, echo canceller included, from files.
///
/// Thread use: played() on the playback thread, add_to() on the capture thread (an SPSC ring
/// between them); both are paced in real time, so the ring's prefill sets the delay.
class EchoSimulator {
public:
    struct Config {
        float gain_db = -6.0f;    ///< echo level relative to what is played
        float delay_ms = 40.0f;   ///< playback-to-capture delay: device buffers plus flight time
        float rt60_ms = 250.0f;   ///< room reverberation time
        float drive = 2.0f;       ///< loudspeaker soft clipping, tanh(drive x) / drive; 0 = linear
        std::uint32_t seed = 1;   ///< of the reverberation tail
    };

    EchoSimulator(Config config, int playback_rate, int capture_rate);

    /// Playback thread: what the loudspeaker plays.
    void played(std::span<const float> samples);
    /// Capture thread: adds the echo that arrives with these capture samples.
    void add_to(std::span<float> capture);

    /// The room's impulse response at the capture rate (direct path first), for tests.
    [[nodiscard]] const std::vector<float>& impulse_response() const noexcept { return rir_; }

private:
    Config cfg_;
    float gain_;
    Resampler resampler_;
    std::vector<float> resampled_;
    SpscRing<float> ring_;
    std::vector<float> rir_;
    std::vector<float> history_;  ///< the last rir_.size() - 1 inputs, oldest first
    std::vector<float> block_;
};

}  // namespace ee
