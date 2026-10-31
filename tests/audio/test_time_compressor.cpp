#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <thread>
#include <vector>

#include "core/audio/audio_io.hpp"
#include "core/audio/playback.hpp"
#include "core/audio/time_compressor.hpp"
#include "support/signals.hpp"
#include "support/test_context.hpp"

namespace ee {
namespace {

constexpr int kRate = 24000;

/// Plays `input` through the compressor in 10 ms blocks, as the device callback would: input is
/// staged ahead of each block. `speed_for` gives the speed for each block from the output so far.
template <class SpeedFor>
std::vector<float> play(TimeCompressor& c, const std::vector<float>& input, SpeedFor speed_for,
                        std::uint64_t limit = TimeCompressor::kNoLimit) {
    std::vector<float> out;
    std::vector<float> block(240);
    std::size_t fed = 0;
    while (true) {
        const std::size_t ahead = c.lookahead(block.size());
        if (c.staged() < ahead && fed < input.size()) {
            const std::span<float> space = c.input_space(ahead - c.staged());
            const std::size_t n = std::min(space.size(), input.size() - fed);
            std::copy_n(input.data() + fed, n, space.data());
            c.commit(n);
            fed += n;
        }
        const std::size_t n = c.produce(block, speed_for(out.size()), limit);
        if (n == 0 && fed >= input.size()) break;
        out.insert(out.end(), block.begin(), block.begin() + static_cast<std::ptrdiff_t>(n));
    }
    return out;
}

double max_step(const std::vector<float>& x, std::size_t begin, std::size_t end) {
    double m = 0.0;
    for (std::size_t i = begin + 1; i < std::min(end, x.size()); ++i) m = std::max(m, std::abs(static_cast<double>(x[i]) - x[i - 1]));
    return m;
}

TEST(TimeCompressor, PassesSamplesThroughUntouchedAtSpeedOne) {
    const std::vector<float> input = test::buzz(150.0, 1.0, kRate);
    TimeCompressor c(kRate);
    const std::vector<float> out = play(c, input, [](std::size_t) { return 1.0; });
    EXPECT_EQ(out, input);
    EXPECT_EQ(c.position(), input.size());
    EXPECT_TRUE(c.idle());
}

// 1.3x: the same 220 Hz in about 1/1.3 of the time (the speed ramps up first, over ~0.2 s).
TEST(TimeCompressor, PlaysFasterWithoutChangingThePitch) {
    const std::vector<float> input = test::sine(220.0, 3.0, kRate);
    TimeCompressor c(kRate);
    const std::vector<float> out = play(c, input, [](std::size_t) { return 1.3; });
    const double seconds = static_cast<double>(out.size()) / kRate;
    EXPECT_GT(seconds, 3.0 / 1.3);
    EXPECT_LT(seconds, 3.0 / 1.3 + 0.1);
    EXPECT_NEAR(test::zero_crossing_hz(out, kRate, kRate / 2, out.size() - kRate / 4), 220.0, 1.0);
    // Segments are joined where the waveform matches: no step bigger than the sine's own.
    EXPECT_LT(max_step(out, 0, out.size()), 1.2 * max_step(input, 0, input.size()));
    EXPECT_EQ(c.position(), input.size());
}

// What follows the limit (the next utterance) plays exactly as written, from its first sample.
TEST(TimeCompressor, NeverCompressesPastTheLimit) {
    std::vector<float> input = test::buzz(150.0, 1.0, kRate);
    const std::size_t limit = input.size();
    std::vector<float> next = test::white_noise(0.2, kRate, 0.3f);
    test::append(input, next);
    TimeCompressor c(kRate);
    const std::vector<float> out = play(c, input, [](std::size_t) { return 1.3; }, limit);
    ASSERT_GT(out.size(), next.size());
    EXPECT_LT(out.size(), input.size() - kRate / 10);  // the first part did speed up
    EXPECT_TRUE(std::equal(next.begin(), next.end(), out.end() - static_cast<std::ptrdiff_t>(next.size())));
}

// Back to speed 1, it ramps down and then passes the input through exactly from where it is.
TEST(TimeCompressor, ReturnsToPlainPlaybackSeamlessly) {
    const std::vector<float> input = test::sine(220.0, 3.0, kRate);
    TimeCompressor c(kRate);
    std::size_t plain_from = 0;
    std::uint64_t plain_at = 0;
    const std::vector<float> out = play(c, input, [&](std::size_t produced) {
        if (produced >= static_cast<std::size_t>(kRate) && !c.compressing() && plain_from == 0) {
            plain_from = produced;
            plain_at = c.position();
        }
        return produced < static_cast<std::size_t>(kRate) ? 1.3 : 1.0;
    });
    ASSERT_GT(plain_from, 0u);
    EXPECT_LT(plain_from, static_cast<std::size_t>(kRate * 1.4));  // the ramp down took < 0.4 s
    EXPECT_TRUE(std::equal(out.begin() + static_cast<std::ptrdiff_t>(plain_from), out.end(),
                           input.begin() + static_cast<std::ptrdiff_t>(plain_at)));
    EXPECT_LT(max_step(out, 0, out.size()), 1.2 * max_step(input, 0, input.size()));
}

TEST(TimeCompressor, FlushDropsTheStagedInputAndCountsIt) {
    TimeCompressor c(kRate);
    const std::span<float> space = c.input_space(1000);
    std::fill(space.begin(), space.end(), 0.5f);
    c.commit(space.size());
    c.flush(500);
    EXPECT_EQ(c.position(), 1500u);
    EXPECT_TRUE(c.idle());
}

TEST(RingSink, CatchesUpOnRequestAndCountsWhatHasPlayed) {
    RingSink sink(kRate, 1 << 17);
    const std::vector<float> speech = test::buzz(150.0, 3.0, kRate);
    ASSERT_EQ(sink.write(speech), speech.size());
    std::vector<float> block(240);
    for (int i = 0; i < 50; ++i) EXPECT_EQ(sink.pull(block), block.size());  // 0.5 s as written
    EXPECT_EQ(sink.played(), 12000u);
    sink.set_catch_up(1.3, TimeCompressor::kNoLimit);
    for (int i = 0; i < 100; ++i) EXPECT_EQ(sink.pull(block), block.size());  // 1 s, faster
    EXPECT_GT(sink.played(), 12000u + 27000u);
    EXPECT_EQ(sink.queued(), speech.size() - sink.played());
}

// As live: playback writes and sets the speed on its thread while the device pulls on another.
// Everything written is played or skipped, once.
TEST(RingSink, CatchUpRunsAlongsideTheWriter) {
    RingSink sink(kRate, 1 << 16);
    const std::vector<float> speech = test::buzz(150.0, 2.0, kRate);
    std::atomic<bool> written{false};
    std::thread writer([&] {
        for (std::size_t pos = 0; pos < speech.size(); pos += 2400) {
            const std::size_t n = std::min<std::size_t>(2400, speech.size() - pos);
            sink.set_catch_up(pos % 4800 == 0 ? 1.3 : 1.0, pos + 2 * n);
            std::span<const float> chunk(speech.data() + pos, n);
            while (!chunk.empty()) chunk = chunk.subspan(sink.write(chunk));
        }
        sink.set_catch_up(1.0, TimeCompressor::kNoLimit);
        written = true;
    });
    std::vector<float> block(240);
    while (!written.load() || sink.queued() > 0) sink.pull(block);
    writer.join();
    EXPECT_EQ(sink.played(), speech.size());
}

Frame synth(std::uint64_t utterance, std::vector<float> audio, double src_end) {
    Frame f;
    f.reset(FrameKind::SynthAudio);
    f.utterance = utterance;
    f.sample_rate = kRate;
    f.flags = frame_flags::kClauseEnd | frame_flags::kFinal;
    f.src_start = src_end - 1.0;
    f.src_end = src_end;
    f.audio = std::move(audio);
    return f;
}

// The speaker has started utterance 2 while utterance 1's translation plays: the rest of it plays
// faster, and utterance 2's translation, queued behind, starts as written once 1 has played.
TEST(Playback, CatchesUpWhileNewerSpeechWaits) {
    RingSink sink(kRate, 1 << 18);
    AudioIo io;
    io.playback = &sink;
    test::RecordingContext ctx;
    ctx.services().audio = &io;
    ctx.mutable_params().set("catch_up_speed", "1.3");
    ctx.telemetry().begin_utterance(1);
    ctx.telemetry().begin_utterance(2);
    PlaybackStage stage;
    stage.open(ctx);

    Frame first = synth(1, test::buzz(150.0, 2.0, kRate), 1.0);
    stage.process(first);
    std::vector<float> block(240);
    std::vector<float> out;
    auto play_for = [&](double seconds) {
        for (int i = 0; i < static_cast<int>(seconds * 100); ++i) {
            sink.pull(block);
            out.insert(out.end(), block.begin(), block.end());
            stage.tick();
        }
    };
    // Marks are back-dated to when the sample played, so may precede the telemetry's epoch here.
    auto first_audio = [&](std::uint64_t u) { return ctx.telemetry().milestone_us(u, telemetry::Milestone::FirstAudio); };
    play_for(0.5);
    EXPECT_EQ(sink.played(), 12000u);  // nobody waiting: as written
    EXPECT_NE(first_audio(1), -1);

    io.speech_utterance = 2;  // the speaker talks on
    stage.tick();
    play_for(0.5);
    EXPECT_GT(sink.played(), 12000u + 13000u);

    Frame second = synth(2, std::vector<float>(12000, 0.25f), 3.0);
    stage.process(second);
    EXPECT_EQ(first_audio(2), -1);  // queued, not playing yet
    play_for(1.5);
    // Utterance 1 ended early; utterance 2 then played whole and unchanged.
    const auto start = std::find(out.begin(), out.end(), 0.25f);
    ASSERT_NE(start, out.end());
    EXPECT_LT(start - out.begin(), static_cast<std::ptrdiff_t>(2.0 * kRate) - 3000);
    EXPECT_EQ(std::count(start, out.end(), 0.25f), 12000);
    EXPECT_NE(first_audio(2), -1);
    stage.close();
}

}  // namespace
}  // namespace ee
