#include <gtest/gtest.h>

#include "core/emotion/consistency_stage.hpp"
#include "core/emotion/emotion_stage.hpp"
#include "core/emotion/state_tracker.hpp"
#include "support/signals.hpp"
#include "support/test_context.hpp"

namespace ee {
namespace {

std::vector<Frame> audio_frames(std::uint64_t utterance, const std::vector<float>& audio, bool endpoint = true) {
    std::vector<Frame> out;
    constexpr std::size_t kWindow = 512;
    for (std::size_t pos = 0; pos < audio.size(); pos += kWindow) {
        Frame f;
        f.reset(FrameKind::Audio);
        f.utterance = utterance;
        f.sample_rate = 16000;
        f.stream_pos = static_cast<std::int64_t>(pos);
        f.flags = pos == 0 ? frame_flags::kSpeechStart : 0u;
        f.audio.assign(audio.begin() + static_cast<std::ptrdiff_t>(pos),
                       audio.begin() + static_cast<std::ptrdiff_t>(std::min(audio.size(), pos + kWindow)));
        out.push_back(f);
    }
    if (endpoint) {
        Frame e;
        e.reset(FrameKind::Audio);
        e.utterance = utterance;
        e.sample_rate = 16000;
        e.flags = frame_flags::kEndpoint | frame_flags::kFinal;
        e.src_end = static_cast<double>(audio.size()) / 16000.0;
        out.push_back(e);
    }
    return out;
}

Frame transcript(std::uint64_t utterance, std::string text, bool final, std::vector<Word> words = {}) {
    Frame f;
    f.reset(FrameKind::Transcript);
    f.utterance = utterance;
    f.text = std::move(text);
    f.language = "en";
    f.words = std::move(words);
    f.flags = final ? frame_flags::kFinal : 0u;
    return f;
}

std::vector<float> excited_speech() {
    test::SyllableTrain t;
    t.count = 12;
    t.syllable_s = 0.12;
    t.gap_s = 0.05;
    t.f0_start = 170.0;
    t.f0_end = 250.0;
    t.rolloff = 1.0;
    return test::syllable_train(t, 16000);
}

TEST(EmotionEngineStage, EmitsPartialsThenOneFinalWithEnvelope) {
    test::RecordingContext ctx;
    ctx.mutable_pipeline().sample_rate = 16000;
    ctx.telemetry().begin_utterance(1);
    EmotionEngineStage stage;
    stage.open(ctx);

    const auto audio = excited_speech();
    for (Frame& f : audio_frames(1, audio)) stage.process(f);
    EXPECT_TRUE(ctx.of(FrameKind::Emotion).size() >= 2u);  // partials every 500 ms after the first second
    for (const Frame& f : ctx.of(FrameKind::Emotion)) EXPECT_FALSE(f.is_final());
    EXPECT_GE(ctx.telemetry().milestone_us(1, telemetry::Milestone::EmotionFinal), 0);

    Frame text = transcript(1, "This is amazing!", true);
    stage.process(text);
    const auto emotions = ctx.of(FrameKind::Emotion);
    ASSERT_FALSE(emotions.empty());
    const Frame& final = emotions.back();
    ASSERT_TRUE(final.is_final());
    EXPECT_EQ(final.utterance, 1u);
    EXPECT_GT(final.emotion.vad.a, 0.3f);  // excited delivery
    EXPECT_GT(final.emotion.vad.v, 0.3f);  // "amazing!" is positive
    EXPECT_GT(final.emotion.confidence, 0.3f);
    EXPECT_NEAR(static_cast<double>(final.envelope.size()), static_cast<double>(audio.size()) / 160.0, 2.0);
    EXPECT_FLOAT_EQ(final.envelope_hop, 0.01f);
}

TEST(EmotionEngineStage, FinalizesWithoutTextWhenNoLexicalModel) {
    test::RecordingContext ctx;
    ctx.mutable_pipeline().sample_rate = 16000;
    ctx.mutable_params().set("lexical", "none");
    EmotionEngineStage stage;
    stage.open(ctx);
    for (Frame& f : audio_frames(7, excited_speech())) stage.process(f);
    const auto emotions = ctx.of(FrameKind::Emotion);
    ASSERT_FALSE(emotions.empty());
    EXPECT_TRUE(emotions.back().is_final());
}

TEST(StateTracker, JoinsTextWithEmotionAndMarksEmphasis) {
    test::RecordingContext ctx;
    ctx.telemetry().begin_utterance(3);
    StateTrackerStage stage;
    stage.open(ctx);

    const std::vector<Word> words = {{"I", 0.00f, 0.10f, 1.0f},   {"can't", 0.12f, 0.40f, 1.0f}, {"believe", 0.42f, 0.80f, 1.0f},
                                     {"you", 0.82f, 0.95f, 1.0f}, {"did", 0.97f, 1.15f, 1.0f},   {"this!", 1.17f, 1.50f, 1.0f}};
    Frame text = transcript(3, "I can't believe you did this!", true, words);
    stage.process(text);
    EXPECT_TRUE(ctx.emitted.empty());  // waits for the emotion

    Frame emotion;
    emotion.reset(FrameKind::Emotion);
    emotion.utterance = 3;
    emotion.flags = frame_flags::kFinal;
    emotion.emotion.vad = prototype(EmotionLabel::Anger);
    emotion.emotion.confidence = 0.87f;
    emotion.envelope.assign(160, -24.0f);
    for (std::size_t i = 0; i < emotion.envelope.size(); ++i) emotion.envelope[i] += static_cast<float>(i % 5) * 0.4f;
    for (std::size_t i = 50; i < 70; ++i) emotion.envelope[i] = -14.0f;
    emotion.envelope_hop = 0.01f;
    stage.process(emotion);

    const auto utterances = ctx.of(FrameKind::Utterance);
    ASSERT_EQ(utterances.size(), 1u);
    const Frame& u = utterances[0];
    EXPECT_EQ(u.text, "I can't believe you did this!");
    EXPECT_EQ(u.emotion.label, EmotionLabel::Anger);
    EXPECT_FLOAT_EQ(u.emotion.confidence, 0.87f);
    ASSERT_EQ(u.emphasis.size(), 1u);
    EXPECT_EQ(u.words[u.emphasis[0]].text, "believe");
    EXPECT_GE(ctx.telemetry().milestone_us(3, telemetry::Milestone::StateReady), 0);
}

TEST(StateTracker, CanRunWithoutAnEmotionEngine) {
    test::RecordingContext ctx;
    ctx.mutable_params().set("wait_for_emotion", "false");
    StateTrackerStage stage;
    stage.open(ctx);
    Frame text = transcript(1, "hello there", true);
    stage.process(text);
    ASSERT_EQ(ctx.of(FrameKind::Utterance).size(), 1u);
    EXPECT_TRUE(ctx.of(FrameKind::Utterance)[0].emphasis.empty());
}

Frame synth_audio(const std::vector<float>& audio, std::uint32_t flags, Vad source) {
    Frame f;
    f.reset(FrameKind::SynthAudio);
    f.utterance = 1;
    f.sample_rate = 16000;
    f.flags = flags;
    f.language = "hi";
    f.text = "clause";
    f.emotion.vad = source;
    f.audio = audio;
    return f;
}

TEST(EmotionConsistency, ScoresClausesAgainstTheSourceEmotion) {
    test::RecordingContext ctx;
    ctx.mutable_pipeline().sample_rate = 16000;
    EmotionConsistencyStage stage;
    stage.open(ctx);

    // Neutral calibration render from the TTS voice.
    test::SyllableTrain neutral;
    neutral.count = 10;
    Frame calibration = synth_audio(test::syllable_train(neutral, 16000), frame_flags::kCalibration, {});
    stage.process(calibration);
    EXPECT_TRUE(ctx.emitted.empty());

    const Vad angry = prototype(EmotionLabel::Anger);
    Frame clause = synth_audio(excited_speech(), frame_flags::kClauseEnd | frame_flags::kFinal, angry);
    stage.process(clause);
    const auto feedback = ctx.of(FrameKind::Feedback);
    ASSERT_EQ(feedback.size(), 1u);
    const Frame& fb = feedback[0];
    EXPECT_NEAR(fb.score, 1.0f - distance(angry, fb.emotion.vad) / kVadDiameter, 1e-5f);
    EXPECT_EQ(fb.delta, fb.emotion.vad - angry);
    EXPECT_GT(fb.emotion.vad.a, 0.3f);           // the excited render reads as aroused
    EXPECT_LT(fb.axis_confidence.v, 0.1f);        // valence is not measurable acoustically in v1
    EXPECT_EQ(ctx.telemetry().ecs().count, 1u);
}

}  // namespace
}  // namespace ee
