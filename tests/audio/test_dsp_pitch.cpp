#include <gtest/gtest.h>

#include "core/audio/dsp.hpp"
#include "core/audio/pitch.hpp"
#include "core/audio/speaker.hpp"
#include "support/signals.hpp"

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

}  // namespace
}  // namespace ee
