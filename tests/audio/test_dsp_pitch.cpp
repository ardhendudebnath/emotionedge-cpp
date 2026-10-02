#include <gtest/gtest.h>

#include <cmath>

#include "core/audio/dsp.hpp"
#include "core/audio/pitch.hpp"
#include "core/audio/speaker.hpp"
#include "support/signals.hpp"
#include "support/test_context.hpp"

namespace ee {
namespace {

TEST(Dsp, RmsOfFullScaleSineIsMinusThreeDb) {
    EXPECT_NEAR(rms_dbfs(test::sine(1000.0, 0.1, 16000, 1.0f)), -3.01f, 0.05f);
    EXPECT_FLOAT_EQ(rms_dbfs(test::silence(0.1, 16000)), -120.0f);
}

TEST(Dsp, SpectralTiltRisesWithBrightness) {
    const float dull = hf_ratio_db(test::sine(150.0, 0.2, 16000));
    const float voiced = hf_ratio_db(test::buzz(150.0, 0.2, 16000));
    const float noise = hf_ratio_db(test::white_noise(0.2, 16000, 0.3f));
    EXPECT_LT(dull, voiced);
    EXPECT_LT(voiced, noise);
}

TEST(Dsp, AgcConvergesTowardTargetAndLimits) {
    Agc::Config cfg;
    cfg.target_dbfs = -23.0f;
    cfg.attack_s = 1.0f;
    cfg.release_s = 3.0f;
    Agc agc(cfg, 16000);
    auto quiet = test::buzz(160.0, 12.0, 16000, 0.02f);  // around -40 dBFS
    for (std::size_t pos = 0; pos + 320 <= quiet.size(); pos += 320) {
        agc.process(std::span<float>(quiet.data() + pos, 320));
    }
    const std::size_t tail = quiet.size() - 16000;
    EXPECT_NEAR(rms_dbfs(std::span<const float>(quiet.data() + tail, 16000)), -23.0f, 2.0f);
    EXPECT_LE(agc.gain_db(), cfg.max_gain_db);

    std::vector<float> hot(320, 3.0f);
    agc.process(hot);
    EXPECT_LE(peak_abs(hot), cfg.limit + 1e-6f);
}

TEST(Dsp, SoftLimitIsTransparentBelowTheKnee) {
    EXPECT_FLOAT_EQ(soft_limit(0.5f, 0.95f), 0.5f);
    EXPECT_LT(soft_limit(2.0f, 0.95f), 0.95f);
    EXPECT_GT(soft_limit(-2.0f, 0.95f), -0.95f);
}

class YinTones : public ::testing::TestWithParam<double> {};

TEST_P(YinTones, TracksPureAndHarmonicTones) {
    const double f0 = GetParam();
    for (const auto& signal : {test::sine(f0, 0.5, 16000), test::buzz(f0, 0.5, 16000)}) {
        PitchConfig cfg;
        const auto frames = track_pitch(signal, cfg);
        int voiced = 0;
        for (std::size_t i = 2; i + 6 < frames.size(); ++i) {
            ASSERT_TRUE(frames[i].voiced) << "frame " << i;
            EXPECT_NEAR(frames[i].f0_hz, f0, f0 * 0.01);
            ++voiced;
        }
        EXPECT_GT(voiced, 30);
    }
}

INSTANTIATE_TEST_SUITE_P(VoiceRange, YinTones, ::testing::Values(90.0, 150.0, 220.0, 330.0));

TEST(Yin, NoiseAndSilenceAreUnvoiced) {
    PitchConfig cfg;
    int voiced = 0;
    for (const auto& f : track_pitch(test::white_noise(0.5, 16000, 0.3f), cfg)) voiced += f.voiced ? 1 : 0;
    EXPECT_LT(voiced, 5);
    for (const auto& f : track_pitch(test::silence(0.2, 16000), cfg)) EXPECT_FALSE(f.voiced);
}

TEST(Speaker, PitchVoicePrintCarriesMedianF0) {
    PitchSpeakerEncoder encoder;
    const auto print = encoder.embed(test::buzz(180.0, 1.0, 16000), 16000);
    EXPECT_NEAR(voice_print_f0(print), 180.0f, 3.0f);
    EXPECT_FLOAT_EQ(voice_print_f0(SpeakerEmbedding{}), 0.0f);
}

// A voice print along `axis`, optionally tilted toward `other` by `amount`.
SpeakerEmbedding voice(std::size_t axis, std::size_t other = 0, float amount = 0.0f) {
    SpeakerEmbedding p{};
    p[axis] = 1.0f;
    p[other] += amount;
    return p;
}

// Speaker-aware barge-in: the interpreted speaker talking on is not an interruption, nor is the
// translation's own voice coming back through the microphone; another voice is.
TEST(VoiceGate, TellsTheSpeakerTheEchoAndOthersApart) {
    VoiceGate gate({.threshold = 0.35f, .echo_threshold = 0.6f, .min_enrolled = 1});
    const SpeakerEmbedding speaker = voice(0), other = voice(1), output = voice(2);
    EXPECT_EQ(gate.judge(speaker), VoiceGate::Verdict::Unknown);  // nobody enrolled yet
    gate.enroll(speaker);
    float to_speaker = 0.0f;
    EXPECT_EQ(gate.judge(voice(0, 1, 0.5f), &to_speaker), VoiceGate::Verdict::Speaker);
    EXPECT_NEAR(to_speaker, 1.0f / std::sqrt(1.25f), 1e-5f);
    EXPECT_EQ(gate.judge(other), VoiceGate::Verdict::Other);
    EXPECT_EQ(gate.judge(output), VoiceGate::Verdict::Other);  // no output heard yet
    gate.hear_output(output);
    EXPECT_EQ(gate.judge(output), VoiceGate::Verdict::Echo);
    EXPECT_EQ(gate.judge(SpeakerEmbedding{}), VoiceGate::Verdict::Unknown);  // too short to embed

    gate.enroll(other);  // another voice's utterance leaves the reference alone
    EXPECT_EQ(gate.judge(other), VoiceGate::Verdict::Other);
    EXPECT_EQ(gate.judge(speaker), VoiceGate::Verdict::Speaker);
}

// A reference from one or two utterances rejects the speaker far more often, so the gate judges
// nobody until min_enrolled utterances in the speaker's voice are in.
TEST(VoiceGate, ArmsOnlyOnceTheSpeakerIsEnrolled) {
    VoiceGate gate({.threshold = 0.25f, .min_enrolled = 3});
    gate.enroll(voice(0));
    gate.enroll(voice(1));  // another voice: not enrolled
    gate.enroll(voice(0, 1, 0.2f));
    EXPECT_FALSE(gate.has_reference());
    EXPECT_EQ(gate.judge(voice(1)), VoiceGate::Verdict::Unknown);
    gate.enroll(voice(0, 2, 0.2f));
    EXPECT_TRUE(gate.has_reference());
    EXPECT_EQ(gate.judge(voice(1)), VoiceGate::Verdict::Other);
    EXPECT_EQ(gate.judge(voice(0)), VoiceGate::Verdict::Speaker);
}

TEST(VoiceGate, RefinesTheReferenceWithTheSpeakersOwnUtterances) {
    VoiceGate gate({.threshold = 0.35f, .min_enrolled = 1});
    gate.enroll(voice(0));
    const SpeakerEmbedding later = voice(0, 1, 0.75f);  // cos 0.8: the same voice, another day
    float before = 0.0f, after = 0.0f;
    (void)gate.judge(later, &before);
    gate.enroll(later);
    (void)gate.judge(later, &after);
    EXPECT_NEAR(before, 0.8f, 1e-5f);
    EXPECT_GT(after, before);
    EXPECT_GT(voice_similarity(voice(0), voice(0, 1, 0.75f)), 0.79f);
    EXPECT_EQ(voice_similarity(voice(0), SpeakerEmbedding{}), 0.0f);
}

TEST(SpeakerStage, BargeInNeedsVoicePrintsNotPitchStatistics) {
    test::RecordingContext ctx;
    ctx.mutable_params().set("barge_in", "true");  // with the default pitch encoder
    SpeakerStage stage;
    EXPECT_THROW(stage.open(ctx), ConfigError);
}

}  // namespace
}  // namespace ee
