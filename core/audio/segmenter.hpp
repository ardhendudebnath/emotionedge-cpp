#pragma once

#include <cstdint>
#include <deque>
#include <memory>
#include <vector>

#include "core/audio/speech_detector.hpp"
#include "core/runtime/stage.hpp"

namespace ee {

struct SegmenterConfig {
    float threshold_on = 0.5f;    ///< speech probability that starts / sustains speech
    float threshold_off = 0.35f;  ///< below this counts toward the hangover
    int min_speech_ms = 90;       ///< onset must last this long (rejects clicks)
    int hangover_ms = 160;        ///< blueprint 1.3: endpointing, 160 ms hangover
    int preroll_ms = 200;         ///< audio kept from before the onset
    float max_utterance_s = 15.0f;
};

/// The speech/silence state machine behind the segmenter, one decision per detector window.
class EndpointTracker {
public:
    enum class Action {
        Silence,   ///< not speech (window may still be pre-roll)
        Start,     ///< onset confirmed: emit pre-roll, then this window
        Continue,  ///< inside an utterance
        End,       ///< endpoint: this window closes the utterance
    };

    EndpointTracker(SegmenterConfig config, double window_seconds);
    Action update(float probability);
    void reset();

    [[nodiscard]] bool in_speech() const noexcept { return in_speech_; }
    /// Windows since the last window at or above threshold_off (the hangover so far).
    [[nodiscard]] int trailing_quiet_windows() const noexcept { return quiet_run_; }
    [[nodiscard]] int onset_windows() const noexcept { return onset_run_; }

private:
    SegmenterConfig cfg_;
    int min_speech_windows_;
    int hangover_windows_;
    int max_windows_;
    bool in_speech_ = false;
    int onset_run_ = 0;
    int quiet_run_ = 0;
    int utterance_windows_ = 0;
};

/// Stage 1.3 "VAD & Segmenter". Consumes the front-end's 20 ms frames, decides speech/silence
/// per detector window, and emits utterance-tagged Audio frames (with pre-roll) followed by an
/// endpoint marker once the hangover expires. Also flags barge-in when speech starts while
/// translated audio is playing.
class SegmenterStage final : public IStage {
public:
    void open(StageContext& ctx) override;
    void process(Frame& frame) override;
    void close() override;

private:
    struct Window {
        std::int64_t pos = 0;
        TimePoint origin{};
        std::vector<float> samples;
    };
    void handle_window(float probability);
    void emit_window(const Window& w, bool first);
    void end_utterance();
    [[nodiscard]] TimePoint origin_at(std::int64_t pos) const;

    StageContext* ctx_ = nullptr;
    std::unique_ptr<ISpeechDetector> detector_;
    std::unique_ptr<EndpointTracker> tracker_;
    int rate_ = 16000;
    std::size_t window_ = 480;
    std::size_t preroll_windows_ = 6;

    std::vector<float> pending_;
    std::int64_t pending_pos_ = 0;
    std::int64_t last_frame_pos_ = 0;
    TimePoint last_frame_origin_{};
    bool have_origin_ = false;

    std::deque<Window> preroll_;
    std::vector<Window> spare_;
    Window current_;

    std::uint64_t utterance_ = 0;
    std::int64_t utterance_start_pos_ = 0;
    std::int64_t speech_end_pos_ = 0;
    std::uint32_t seq_ = 0;
};

}  // namespace ee
