// Final passes started inside the endpoint hangover (SpeculationTracker).
#include <gtest/gtest.h>

#include <cstdint>

#include "core/runtime/frame.hpp"
#include "core/runtime/speculation.hpp"

namespace ee {
namespace {

Frame window(bool quiet, std::size_t samples = 512) {
    Frame f;
    f.reset(FrameKind::Audio);
    f.audio.assign(samples, 0.0f);
    f.flags = quiet ? frame_flags::kQuiet : 0u;
    return f;
}

using Event = SpeculationTracker::Event;

TEST(SpeculationTracker, StartsOnceEnoughQuietHasArrived) {
    SpeculationTracker t(1024);  // 64 ms at 16 kHz
    EXPECT_EQ(t.on_audio(window(false)), Event::None);
    EXPECT_EQ(t.on_audio(window(true)), Event::None);   // 32 ms of quiet
    EXPECT_EQ(t.on_audio(window(true)), Event::Start);  // 64 ms: run the final pass now
    EXPECT_EQ(t.on_audio(window(true)), Event::None);   // once per quiet run
    EXPECT_TRUE(t.holds());                              // the endpoint can use it
}

TEST(SpeculationTracker, SpeechResumingDropsTheEarlyResult) {
    SpeculationTracker t(1024);
    t.on_audio(window(true));
    ASSERT_EQ(t.on_audio(window(true)), Event::Start);
    EXPECT_EQ(t.on_audio(window(false)), Event::Cancel);  // the speaker went on
    EXPECT_FALSE(t.holds());
    EXPECT_EQ(t.on_audio(window(true)), Event::None);  // a new quiet run counts from zero
    EXPECT_EQ(t.on_audio(window(true)), Event::Start);
    t.reset();  // the next utterance
    EXPECT_FALSE(t.holds());
}

TEST(SpeculationTracker, OffByDefault) {
    SpeculationTracker t;
    for (int i = 0; i < 10; ++i) EXPECT_EQ(t.on_audio(window(true)), Event::None);
    EXPECT_FALSE(t.holds());
}

}  // namespace
}  // namespace ee
