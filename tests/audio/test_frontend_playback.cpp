#include <gtest/gtest.h>

#include "core/audio/audio_io.hpp"
#include "core/audio/frontend.hpp"
#include "core/audio/playback.hpp"
#include "support/signals.hpp"
#include "support/test_context.hpp"

namespace ee {
namespace {

TEST(Frontend, StreamsResampledTwentyMillisecondFrames) {
    BufferSource source(test::sine(500.0, 1.0, 48000, 0.3f), 48000);
    AudioIo io;
    io.capture = &source;
    test::RecordingContext ctx;
    ctx.mutable_pipeline().sample_rate = 16000;
    ctx.services().audio = &io;
    ctx.mutable_params().set("agc.enabled", "false");

    FrontendStage stage;
    stage.open(ctx);
    int ticks = 0;
    while (!ctx.finished && ticks < 1000) {
        stage.tick();
        ++ticks;
    }
    ASSERT_TRUE(ctx.finished);
    std::int64_t expected_pos = 0;
    std::size_t total = 0;
    for (const Frame& f : ctx.emitted) {
        EXPECT_EQ(f.kind, FrameKind::Audio);
        EXPECT_EQ(f.sample_rate, 16000);
        EXPECT_EQ(f.stream_pos, expected_pos);
        EXPECT_LE(f.audio.size(), 320u);
        expected_pos += static_cast<std::int64_t>(f.audio.size());
        total += f.audio.size();
    }
    EXPECT_EQ(total, 16000u);
    EXPECT_GE(ctx.emitted.size(), 50u);
}

TEST(Frontend, RequiresACaptureSource) {
    test::RecordingContext ctx;
    FrontendStage stage;
    EXPECT_THROW(stage.open(ctx), ConfigError);
}

TEST(ChunkJoiner, ConcatenatesWithinAClauseAndCrossfadesBetweenClauses) {
    ChunkJoiner joiner(4);
    std::vector<float> out;
    joiner.push(std::vector<float>(10, 1.0f), false, out);
    EXPECT_EQ(out.size(), 10u);
    joiner.push(std::vector<float>(10, 1.0f), true, out);  // clause end: hold 4 samples back
    EXPECT_EQ(out.size(), 16u);
    joiner.push(std::vector<float>(10, 1.0f), false, out);  // overlaps the held tail
    EXPECT_EQ(out.size(), 26u);                             // 30 in, 4 overlapped
    joiner.finish(out);
    EXPECT_EQ(out.size(), 26u);
    for (float v : out) EXPECT_NEAR(v, 1.0f, 0.42f);  // equal-power fade never dips below cos(45°)·2
}

Frame synth_chunk(std::uint64_t utterance, std::size_t samples, std::uint32_t flags, double src_end) {
    Frame f;
    f.reset(FrameKind::SynthAudio);
    f.utterance = utterance;
    f.sample_rate = 24000;
    f.flags = flags;
    f.src_start = src_end - 1.0;
    f.src_end = src_end;
    f.audio.assign(samples, 0.1f);
    return f;
}

TEST(Playback, OfflinePlacesUtterancesOnTheTimeline) {
    TimelineSink timeline(24000);
    AudioIo io;
    io.timeline = &timeline;
    test::RecordingContext ctx;
    ctx.services().audio = &io;
    ctx.telemetry().begin_utterance(1);
    PlaybackStage stage;
    stage.open(ctx);

    Frame a = synth_chunk(1, 2400, frame_flags::kClauseEnd, 2.0);
    stage.process(a);
    Frame b = synth_chunk(1, 2400, frame_flags::kClauseEnd | frame_flags::kFinal, 2.0);
    stage.process(b);
    Frame c = synth_chunk(2, 2400, frame_flags::kClauseEnd | frame_flags::kFinal, 2.2);  // source overlaps
    stage.process(c);

    const auto playouts = ctx.of(FrameKind::Playout);
    ASSERT_EQ(playouts.size(), 2u);
    EXPECT_NEAR(playouts[0].out_start, 2.0 + 0.735, 1e-9);  // source end + nominal latency
    EXPECT_NEAR(playouts[0].out_end - playouts[0].out_start, (4800.0 - 120.0) / 24000.0, 1e-3);
    EXPECT_NEAR(playouts[1].out_start, playouts[0].out_end + 0.15, 1e-3);  // never overlaps
    EXPECT_GE(ctx.telemetry().milestone_us(1, telemetry::Milestone::FirstAudio), 0);
}

TEST(Playback, BargeInCancelsQueuedTranslation) {
    RingSink sink(24000, 1 << 16);
    AudioIo io;
    io.playback = &sink;
    test::RecordingContext ctx;
    ctx.services().audio = &io;
    PlaybackStage stage;
    stage.open(ctx);

    Frame a = synth_chunk(1, 4800, 0, 1.0);
    stage.process(a);
    EXPECT_GT(sink.queued(), 0u);
    EXPECT_TRUE(io.playback_active.load());

    Frame barge;
    barge.reset(FrameKind::Control);
    barge.flags = frame_flags::kBargeIn;
    barge.utterance = 2;
    stage.process(barge);
    EXPECT_FALSE(io.playback_active.load());
    Frame late = synth_chunk(1, 4800, frame_flags::kFinal, 1.0);  // stale audio of the cancelled utterance
    stage.process(late);
    std::vector<float> drained(1 << 16);
    EXPECT_EQ(sink.pull(drained), 0u);  // flushed, and nothing new was queued
}

// Adaptive pacing (4.2) reads how far playout runs behind: the queued audio live, the timeline's
// next free slot offline.
TEST(Playback, PublishesItsBacklogForPacing) {
    RingSink sink(24000, 1 << 16);
    AudioIo live;
    live.playback = &sink;
    test::RecordingContext ctx;
    ctx.services().audio = &live;
    PlaybackStage stage;
    stage.open(ctx);
    Frame a = synth_chunk(1, 24000, 0, 1.0);  // 1 s, past the jitter buffer
    stage.process(a);
    EXPECT_NEAR(live.playout_delay(1.0), 1.0, 0.02);
    std::vector<float> played(12000);
    EXPECT_EQ(sink.pull(played), 12000u);
    stage.tick();
    EXPECT_NEAR(live.playout_delay(1.0), 0.5, 0.02);

    TimelineSink timeline(24000);
    AudioIo offline;
    offline.timeline = &timeline;
    test::RecordingContext octx;
    octx.services().audio = &offline;
    octx.telemetry().begin_utterance(1);
    PlaybackStage ostage;
    ostage.open(octx);
    Frame b = synth_chunk(1, 48000, frame_flags::kClauseEnd | frame_flags::kFinal, 2.0);
    ostage.process(b);
    // 2 s placed from 2.735 s: the timeline is busy until 4.735 s, plus the 0.15 s gap. An
    // utterance whose source ended at 3.0 s would start 1.15 s after its nominal 3.735 s.
    EXPECT_NEAR(offline.playout_delay(3.0), 1.15, 0.01);
    EXPECT_EQ(offline.playout_delay(10.0), 0.0);
}

}  // namespace
}  // namespace ee
