#pragma once

#include <cstddef>

#include "core/runtime/frame.hpp"

namespace ee {

/// Final passes started inside the endpoint hangover. The segmenter declares the endpoint only
/// after `hangover_ms` of quiet (160 ms), so a stage that waits for it adds its own final pass
/// on top. Once `speculate_ms` of quiet utterance audio (frame_flags::kQuiet) has arrived in a
/// row, a stage runs its final pass on the audio so far and keeps the result. If the quiet
/// lasts to the endpoint, the result is used as it stands; if speech resumes, it is dropped.
class SpeculationTracker {
public:
    enum class Event {
        None,
        Start,   ///< run the final pass now and keep its result
        Cancel,  ///< speech resumed: the kept result no longer holds
    };

    SpeculationTracker() = default;
    /// after_samples: quiet audio before a final pass starts early; 0 = never.
    explicit SpeculationTracker(std::size_t after_samples) : after_(after_samples) {}

    /// One audio frame of the utterance, in order.
    Event on_audio(const Frame& f) noexcept {
        if (after_ == 0) return Event::None;
        if (!f.has(frame_flags::kQuiet)) {
            quiet_ = 0;
            if (!kept_) return Event::None;
            kept_ = false;
            return Event::Cancel;
        }
        quiet_ += f.audio.size();
        if (kept_ || quiet_ < after_) return Event::None;
        kept_ = true;
        return Event::Start;
    }

    /// At the endpoint: a result computed early still holds (no speech since).
    [[nodiscard]] bool holds() const noexcept { return kept_; }
    void reset() noexcept {
        quiet_ = 0;
        kept_ = false;
    }

private:
    std::size_t after_ = 0;
    std::size_t quiet_ = 0;
    bool kept_ = false;
};

}  // namespace ee
