#include <gtest/gtest.h>

#include "core/audio/audio_io.hpp"
#include "core/audio/segmenter.hpp"
#include "support/signals.hpp"
#include "support/test_context.hpp"

namespace ee {
namespace {

using Action = EndpointTracker::Action;

TEST(EndpointTracker, NeedsASustainedOnsetAndAFullHangover) {
    SegmenterConfig cfg;  // 32 ms windows: onset 3 windows, hangover 5 windows
    EndpointTracker t(cfg, 0.032);
    EXPECT_EQ(t.update(0.9f), Action::Silence);
    EXPECT_EQ(t.update(0.1f), Action::Silence);  // a click resets the onset
    EXPECT_EQ(t.update(0.9f), Action::Silence);
    EXPECT_EQ(t.update(0.9f), Action::Silence);
    EXPECT_EQ(t.update(0.9f), Action::Start);
    EXPECT_EQ(t.update(0.4f), Action::Continue);  // between thresholds: still speech
    for (int i = 0; i < 4; ++i) EXPECT_EQ(t.update(0.1f), Action::Continue);
    EXPECT_EQ(t.update(0.6f), Action::Continue);  // speech resumes: hangover resets
    for (int i = 0; i < 4; ++i) EXPECT_EQ(t.update(0.1f), Action::Continue);
    EXPECT_EQ(t.update(0.1f), Action::End);  // 5 quiet windows = 160 ms
    EXPECT_FALSE(t.in_speech());
}

TEST(EndpointTracker, SplitsOverlongUtterances) {
    SegmenterConfig cfg;
    cfg.max_utterance_s = 1.0f;
    EndpointTracker t(cfg, 0.032);
    int ends = 0;
    for (int i = 0; i < 200; ++i) ends += t.update(0.9f) == Action::End ? 1 : 0;
    EXPECT_GE(ends, 5);
}

struct Scene {
    std::vector<float> audio;
    std::vector<std::pair<double, double>> speech;  // [start, end] seconds
};

Scene two_utterances() {
    Scene s;
    constexpr int kRate = 16000;
    test::append(s.audio, test::silence(0.5, kRate));
    s.speech.push_back({0.5, 1.5});
    test::append(s.audio, test::buzz(140.0, 1.0, kRate, 0.2f));
    test::append(s.audio, test::silence(0.6, kRate));
    s.speech.push_back({2.1, 2.9});
    test::append(s.audio, test::buzz(200.0, 0.8, kRate, 0.2f));
    test::append(s.audio, test::silence(0.5, kRate));
    return s;
}

void feed(SegmenterStage& stage, const std::vector<float>& audio, TimePoint t0 = Clock::now()) {
    constexpr std::size_t kFrame = 320;
    for (std::size_t pos = 0; pos < audio.size(); pos += kFrame) {
        Frame f;
        f.reset(FrameKind::Audio);
        f.stream_pos = static_cast<std::int64_t>(pos);
        f.sample_rate = 16000;
        f.t_origin = add_seconds(t0, static_cast<double>(pos) / 16000.0);
        f.audio.assign(audio.begin() + static_cast<std::ptrdiff_t>(pos),
                       audio.begin() + static_cast<std::ptrdiff_t>(std::min(audio.size(), pos + kFrame)));
        stage.process(f);
    }
}

TEST(Segmenter, FindsUtterancesWithPrerollAndHangover) {
    test::RecordingContext ctx;
    ctx.mutable_pipeline().sample_rate = 16000;
    SegmenterStage stage;
    stage.open(ctx);
    const Scene scene = two_utterances();
    feed(stage, scene.audio);
    stage.close();

    std::vector<Frame> endpoints;
    std::vector<Frame> starts;
    for (const Frame& f : ctx.emitted) {
        ASSERT_EQ(f.kind, FrameKind::Audio);
        if (f.has(frame_flags::kEndpoint)) endpoints.push_back(f);
        if (f.has(frame_flags::kSpeechStart)) starts.push_back(f);
    }
    ASSERT_EQ(endpoints.size(), 2u);
    ASSERT_EQ(starts.size(), 2u);
    for (std::size_t u = 0; u < 2; ++u) {
        const auto [speech_start, speech_end] = scene.speech[u];
        EXPECT_EQ(endpoints[u].utterance, u + 1);
        // Pre-roll reaches back up to 200 ms before the onset.
        EXPECT_LE(endpoints[u].src_start, speech_start);
        EXPECT_GE(endpoints[u].src_start, speech_start - 0.25);
        // Speech end is found within one window...
        EXPECT_NEAR(endpoints[u].src_end, speech_end, 0.04);
        // ...and the endpoint fires after the 160 ms hangover.
        const double endpoint_time = static_cast<double>(endpoints[u].stream_pos) / 16000.0;
        EXPECT_NEAR(endpoint_time - endpoints[u].src_end, 0.16, 0.035);
    }
    // Every audio frame of an utterance precedes its endpoint and carries its id.
    std::uint64_t current = 0;
    for (const Frame& f : ctx.emitted) {
        if (f.has(frame_flags::kSpeechStart)) current = f.utterance;
        EXPECT_EQ(f.utterance, current);
    }
    auto& telemetry = ctx.telemetry();
    EXPECT_GE(telemetry.milestone_us(1, telemetry::Milestone::SpeechEnd), 0);
    EXPECT_GE(telemetry.milestone_us(2, telemetry::Milestone::Endpoint), 0);
}

TEST(Segmenter, EndOfStreamClosesAnOpenUtterance) {
    test::RecordingContext ctx;
    ctx.mutable_pipeline().sample_rate = 16000;
    SegmenterStage stage;
    stage.open(ctx);
    std::vector<float> audio = test::silence(0.3, 16000);
    test::append(audio, test::buzz(150.0, 0.5, 16000, 0.2f));  // speech runs to the end
    feed(stage, audio);
    stage.close();
    ASSERT_FALSE(ctx.emitted.empty());
    EXPECT_TRUE(ctx.emitted.back().has(frame_flags::kEndpoint));
    EXPECT_NEAR(ctx.emitted.back().src_end, 0.8, 0.04);
}

TEST(Segmenter, FlagsBargeInWhilePlaybackIsActive) {
    test::RecordingContext ctx;
    ctx.mutable_pipeline().sample_rate = 16000;
    AudioIo io;
    io.playback_active = true;
    ctx.services().audio = &io;
    SegmenterStage stage;
    stage.open(ctx);
    std::vector<float> audio = test::silence(0.3, 16000);
    test::append(audio, test::buzz(150.0, 0.4, 16000, 0.2f));
    feed(stage, audio);
    const auto controls = ctx.of(FrameKind::Control);
    ASSERT_EQ(controls.size(), 1u);
    EXPECT_TRUE(controls[0].has(frame_flags::kBargeIn));
    EXPECT_EQ(controls[0].utterance, 1u);
}

}  // namespace
}  // namespace ee
