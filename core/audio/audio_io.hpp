#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <thread>
#include <vector>

#include "core/audio/ring_buffer.hpp"

namespace ee {

/// Where the front-end (1.1) pulls audio from.
class IAudioSource {
public:
    virtual ~IAudioSource() = default;
    [[nodiscard]] virtual int sample_rate() const noexcept = 0;
    /// Reads up to out.size() samples; returns how many were read (0 if none are ready yet).
    virtual std::size_t read(std::span<float> out) = 0;
    /// True once no more samples will ever arrive.
    [[nodiscard]] virtual bool exhausted() const noexcept = 0;
    /// Samples captured but not yet read, used to back-date capture timestamps.
    [[nodiscard]] virtual std::size_t backlog() const noexcept { return 0; }
};

/// Where playback (5.1) sends audio in live runs.
class IAudioSink {
public:
    virtual ~IAudioSink() = default;
    [[nodiscard]] virtual int sample_rate() const noexcept = 0;
    /// Queues samples for playout; returns how many were accepted.
    virtual std::size_t write(std::span<const float> samples) = 0;
    /// Samples queued ahead of the speaker (playout delay = queued / rate).
    [[nodiscard]] virtual std::size_t queued() const noexcept { return 0; }
    /// Drops everything queued (barge-in).
    virtual void flush() {}
    /// True while an utterance is mid-playout, so running dry is a dropout rather than its end.
    virtual void set_streaming(bool streaming) noexcept { (void)streaming; }
};

/// The whole signal is available up front (offline runs and tests).
class BufferSource final : public IAudioSource {
public:
    BufferSource(std::vector<float> samples, int sample_rate);
    [[nodiscard]] int sample_rate() const noexcept override { return rate_; }
    std::size_t read(std::span<float> out) override;
    [[nodiscard]] bool exhausted() const noexcept override { return pos_ >= samples_.size(); }

private:
    std::vector<float> samples_;
    int rate_;
    std::size_t pos_ = 0;
};

/// Fed from another thread through a lock-free ring: the capture callback (T0) or a PacedFeeder.
class RingSource final : public IAudioSource {
public:
    RingSource(int sample_rate, std::size_t capacity);
    /// Producer (T0): never blocks or allocates. Samples that do not fit are counted as dropouts.
    std::size_t push(std::span<const float> samples) noexcept;
    /// Producer: end of stream.
    void close() noexcept { closed_.store(true, std::memory_order_release); }

    [[nodiscard]] int sample_rate() const noexcept override { return rate_; }
    std::size_t read(std::span<float> out) override { return ring_.read(out); }
    [[nodiscard]] bool exhausted() const noexcept override {
        return closed_.load(std::memory_order_acquire) && ring_.read_available() == 0;
    }
    [[nodiscard]] std::size_t backlog() const noexcept override { return ring_.read_available(); }
    [[nodiscard]] std::uint64_t overflow_samples() const noexcept { return overflow_.load(); }

private:
    SpscRing<float> ring_;
    int rate_;
    std::atomic<bool> closed_{false};
    std::atomic<std::uint64_t> overflow_{0};
};

/// Drained by the playback callback (T0).
class RingSink final : public IAudioSink {
public:
    RingSink(int sample_rate, std::size_t capacity);
    /// Consumer (T0): fills `out`, zero-padding on underrun; returns the samples actually played.
    /// Everything handed to the device, silence included, also goes to the played tap.
    std::size_t pull(std::span<float> out) noexcept;
    /// Records what is played, as it is played: the echo canceller's reference (AudioIo::
    /// echo_reference). Set before playback starts; overflow drops reference, never blocks.
    void set_played_tap(SpscRing<float>* tap) noexcept { tap_ = tap; }

    [[nodiscard]] int sample_rate() const noexcept override { return rate_; }
    std::size_t write(std::span<const float> samples) override { return ring_.write(samples); }
    [[nodiscard]] std::size_t queued() const noexcept override { return ring_.read_available(); }
    void flush() override { flush_requested_.store(true, std::memory_order_release); }
    void set_streaming(bool streaming) noexcept override { streaming_.store(streaming, std::memory_order_release); }
    [[nodiscard]] std::uint64_t underrun_samples() const noexcept { return underrun_.load(); }

private:
    SpscRing<float> ring_;
    int rate_;
    SpscRing<float>* tap_ = nullptr;
    std::atomic<bool> flush_requested_{false};
    std::atomic<bool> streaming_{false};
    std::atomic<std::uint64_t> underrun_{0};
};

/// Offline playout: audio is placed at explicit times on a timeline that becomes the output WAV.
class TimelineSink {
public:
    explicit TimelineSink(int sample_rate) : rate_(sample_rate) {}
    [[nodiscard]] int sample_rate() const noexcept { return rate_; }
    /// Mixes `samples` in, starting at `start_seconds`.
    void write_at(double start_seconds, std::span<const float> samples);
    /// End of the latest audio written, in seconds.
    [[nodiscard]] double end_seconds() const noexcept;
    [[nodiscard]] const std::vector<float>& samples() const noexcept { return data_; }

private:
    int rate_;
    std::vector<float> data_;
};

/// Audio endpoints shared by the device layer (T0) and the stages.
struct AudioIo {
    IAudioSource* capture = nullptr;             ///< front-end input
    IAudioSink* playback = nullptr;              ///< live playout
    TimelineSink* timeline = nullptr;            ///< offline playout
    /// What the loudspeaker plays, at the playback rate, written as it is played (a
    /// RingSink::set_played_tap): the echo canceller's reference (1.2).
    SpscRing<float>* echo_reference = nullptr;
    std::atomic<bool> playback_active{false};    ///< translated speech is playing (barge-in)

    // How far the translation runs behind, for the TTS's adaptive pacing (4.2). Playback (5.1)
    // writes both; each run mode uses one.
    /// Live: seconds of translated audio queued ahead of the speaker.
    std::atomic<double> playout_queued_s{0.0};
    /// Offline: an utterance whose source ended at `src_end` starts playing
    /// max(0, playout_free_at_s - src_end) later than the nominal latency.
    std::atomic<double> playout_free_at_s{-1e9};

    /// How long audio for an utterance ending at `src_end` would wait behind earlier translations.
    [[nodiscard]] double playout_delay(double src_end) const noexcept {
        const double offline = playout_free_at_s.load(std::memory_order_relaxed) - src_end;
        return std::max({0.0, playout_queued_s.load(std::memory_order_relaxed), offline});
    }
};

/// Plays a recorded signal into a RingSource in real time, standing in for the capture
/// callback (T0) in benchmarks and live-mode tests.
class PacedFeeder {
public:
    PacedFeeder(RingSource& target, std::vector<float> samples, double speed = 1.0, std::size_t block = 320);
    ~PacedFeeder();
    PacedFeeder(const PacedFeeder&) = delete;
    PacedFeeder& operator=(const PacedFeeder&) = delete;

    /// Called on each block before it is captured (e.g. EchoSimulator::add_to). Set before start().
    void set_capture_hook(std::function<void(std::span<float>)> hook) { hook_ = std::move(hook); }

    void start();
    void stop();
    [[nodiscard]] bool done() const noexcept { return done_.load(); }

private:
    RingSource& target_;
    std::vector<float> samples_;
    double speed_;
    std::size_t block_;
    std::function<void(std::span<float>)> hook_;
    std::thread thread_;
    std::atomic<bool> stop_{false};
    std::atomic<bool> done_{false};
};

/// Pulls a RingSink at real-time rate, standing in for the playback callback (T0) in benchmarks.
/// Everything "played" is recorded, silence included, so the recording has the live timing.
class PacedDrain {
public:
    PacedDrain(RingSink& source, double speed = 1.0, std::size_t block = 480);
    ~PacedDrain();
    PacedDrain(const PacedDrain&) = delete;
    PacedDrain& operator=(const PacedDrain&) = delete;

    /// Called with each block as it is played (e.g. EchoSimulator::played). Set before start().
    void set_play_hook(std::function<void(std::span<const float>)> hook) { hook_ = std::move(hook); }

    void start();
    void stop();
    /// Only valid after stop().
    [[nodiscard]] const std::vector<float>& recording() const noexcept { return recording_; }

private:
    RingSink& source_;
    double speed_;
    std::size_t block_;
    std::function<void(std::span<const float>)> hook_;
    std::thread thread_;
    std::atomic<bool> stop_{false};
    std::vector<float> recording_;
};

}  // namespace ee
