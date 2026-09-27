#include <gtest/gtest.h>

#include "core/emotion/acoustic.hpp"
#include "core/emotion/fusion.hpp"
#include "core/emotion/lexical.hpp"
#include "core/emotion/prosody_features.hpp"
#include "support/signals.hpp"

namespace ee {
namespace {

ProsodySummary analyze(const std::vector<float>& audio) {
    ProsodyTracker tracker(16000);
    tracker.push(audio);
    tracker.flush();
    return summarize(tracker.frames(), tracker.hop_seconds());
}

// ---- prosody features -------------------------------------------------------------------------

TEST(ProsodyTracker, FramesAreCenteredOnTenMillisecondHops) {
    std::vector<float> audio = test::silence(0.5, 16000);
    test::append(audio, test::buzz(150.0, 0.5, 16000));
    ProsodyTracker tracker(16000);
    // Push in odd-sized chunks: framing must not depend on chunking.
    for (std::size_t pos = 0; pos < audio.size(); pos += 333) {
        tracker.push(std::span<const float>(audio.data() + pos, std::min<std::size_t>(333, audio.size() - pos)));
    }
    tracker.flush();
    const auto& frames = tracker.frames();
    EXPECT_NEAR(static_cast<double>(frames.size()), 100.0, 1.0);
    std::size_t first_voiced = frames.size();
    for (std::size_t i = 0; i < frames.size(); ++i) {
        if (frames[i].voiced) {
            first_voiced = i;
            break;
        }
    }
    EXPECT_NEAR(static_cast<double>(first_voiced), 50.0, 3.0);  // voicing starts at 0.5 s
    EXPECT_NEAR(frames[75].f0_hz, 150.0f, 2.0f);
}

TEST(ProsodySummary, MeasuresRatePausesRangeAndContour) {
    test::SyllableTrain t;
    t.count = 10;
    t.f0_start = 120.0;
    t.f0_end = 180.0;
    const ProsodySummary rising = analyze(test::syllable_train(t, 16000));
    const double expected_rate = 10.0 / (10 * (t.syllable_s + t.gap_s) - t.gap_s);
    EXPECT_NEAR(rising.syllable_rate, expected_rate, 0.6);
    EXPECT_GT(rising.pause_ratio, 0.15f);
    // p10..p90 of a linear 120 -> 180 Hz glide: 126 -> 174 Hz, about 5.6 semitones.
    EXPECT_NEAR(rising.f0_range_st, 5.6f, 1.0f);
    EXPECT_GT(rising.final_slope_st_s, 1.0f);

    std::swap(t.f0_start, t.f0_end);
    EXPECT_LT(analyze(test::syllable_train(t, 16000)).final_slope_st_s, -1.0f);
}

// ---- acoustic (prosody) model -----------------------------------------------------------------

TEST(ProsodyEmotionModel, ArousedDeliveryScoresHigherArousal) {
    test::SyllableTrain calm;
    calm.count = 10;
    calm.syllable_s = 0.22;
    calm.gap_s = 0.12;
    calm.f0_start = 140.0;
    calm.f0_end = 130.0;
    calm.rolloff = 1.8;  // dull, lax voice
    test::SyllableTrain aroused;
    aroused.count = 14;
    aroused.syllable_s = 0.12;
    aroused.gap_s = 0.05;
    aroused.f0_start = 170.0;
    aroused.f0_end = 260.0;
    aroused.rolloff = 1.0;  // bright, pressed voice

    ProsodyEmotionModel model;
    const auto low = model.from_summary(analyze(test::syllable_train(calm, 16000)));
    const auto high = model.from_summary(analyze(test::syllable_train(aroused, 16000)));
    ASSERT_TRUE(low.valid);
    ASSERT_TRUE(high.valid);
    EXPECT_LT(low.vad.a, 0.0f);
    EXPECT_GT(high.vad.a, 0.4f);
    EXPECT_GT(high.confidence.a, 0.5f);
    EXPECT_LT(high.confidence.v, 0.1f);  // prosody abstains on valence
}

TEST(ProsodyEmotionModel, RulesMoveTheRightAxes) {
    ProsodyEmotionModel model;
    ProsodySummary neutral;
    neutral.voiced_frames = 60;
    neutral.f0_median_hz = 150.0f;
    neutral.f0_range_st = 4.0f;
    neutral.syllable_rate = 4.0f;
    neutral.hf_db = -13.0f;
    neutral.pause_ratio = 0.2f;
    neutral.jitter = 0.02f;
    const auto n = model.from_summary(neutral);
    EXPECT_NEAR(n.vad.a, 0.0f, 0.05f);
    EXPECT_NEAR(n.vad.d, 0.0f, 0.05f);

    ProsodySummary dominant = neutral;
    dominant.final_slope_st_s = -8.0f;  // falling final contour
    dominant.pause_ratio = 0.05f;       // few hesitations
    EXPECT_GT(model.from_summary(dominant).vad.d, 0.4f);

    ProsodySummary submissive = neutral;
    submissive.final_slope_st_s = 6.0f;  // rising
    submissive.pause_ratio = 0.45f;
    submissive.jitter = 0.05f;
    EXPECT_LT(model.from_summary(submissive).vad.d, -0.4f);

    // Relative loudness only counts once a baseline exists.
    ProsodySummary loud = neutral;
    loud.energy_db = neutral.energy_db + 8.0f;
    EXPECT_NEAR(model.from_summary(loud).vad.a, 0.0f, 0.05f);
    model.calibrate(neutral);
    EXPECT_GT(model.from_summary(loud).vad.a, 0.5f);

    ProsodySummary unvoiced;
    EXPECT_FALSE(model.from_summary(unvoiced).valid);
}

// ---- lexical model ------------------------------------------------------------------------------

TEST(Lexicon, ScoresWordsPhrasesNegationAndPunctuation) {
    LexiconEmotionModel lex;
    const auto happy = lex.estimate("I am so happy!", "en");
    ASSERT_TRUE(happy.valid);
    EXPECT_GT(happy.vad.v, 0.5f);
    EXPECT_GT(happy.vad.a, 0.5f);

    const auto negated = lex.estimate("I'm not happy", "en");
    ASSERT_TRUE(negated.valid);
    EXPECT_LT(negated.vad.v, 0.0f);

    const auto walkthrough = lex.estimate("I can\xE2\x80\x99t believe you did this!", "en");  // curly apostrophe
    ASSERT_TRUE(walkthrough.valid);
    EXPECT_LT(walkthrough.vad.v, 0.0f);
    EXPECT_GT(walkthrough.vad.a, 0.5f);
    EXPECT_GT(walkthrough.vad.d, 0.0f);

    EXPECT_FALSE(lex.estimate("the table is brown", "en").valid);
    EXPECT_FALSE(lex.estimate("I am so happy!", "hi").valid);  // abstains outside English

    const auto shouted = lex.estimate("STOP it NOW", "en");
    const auto spoken = lex.estimate("stop it now", "en");
    EXPECT_GT(shouted.vad.a, spoken.vad.a);
}

TEST(Lexicon, TokenizerNormalizesApostrophesAndPunctuation) {
    EXPECT_EQ(tokenize_for_emotion("Can\xE2\x80\x99t STOP!?"),
              (std::vector<std::string>{"can't", "stop", "!", "?"}));
}

// ---- fusion, smoothing, hysteresis ------------------------------------------------------------

TEST(Fusion, TakesEachAxisFromTheModalityThatKnowsIt) {
    ModalityEstimate acoustic;
    acoustic.valid = true;
    acoustic.vad = {0.0f, 0.8f, 0.5f};
    acoustic.confidence = {0.05f, 0.7f, 0.4f};  // abstains on valence
    ModalityEstimate lexical;
    lexical.valid = true;
    lexical.vad = {-0.6f, 0.2f, 0.2f};
    lexical.confidence = {0.6f, 0.3f, 0.3f};

    Vad axes;
    const EmotionState s = fuse(acoustic, lexical, FusionConfig{}, &axes);
    EXPECT_FLOAT_EQ(s.vad.v, -0.6f);           // acoustic valence was gated out
    EXPECT_GT(s.vad.a, 0.6f);                  // arousal leans on prosody
    EXPECT_NEAR(axes.a, 1.0f - 0.3f * 0.7f, 1e-5f);  // evidence accumulates: 1 - (1-0.7)(1-0.3)
    EXPECT_EQ(s.label, nearest_label(s.vad));

    const EmotionState none = fuse(ModalityEstimate{}, ModalityEstimate{}, FusionConfig{});
    EXPECT_FLOAT_EQ(none.confidence, 0.0f);
    EXPECT_EQ(none.label, EmotionLabel::Neutral);
}

TEST(EmotionSmoother, FirstEstimateIsTakenThenMovesByConfidence) {
    EmotionSmoother smoother(0.5f);
    EmotionState a;
    a.vad = {0.0f, 1.0f, 0.0f};
    a.confidence = 1.0f;
    EXPECT_FLOAT_EQ(smoother.update(a).vad.a, 1.0f);
    EmotionState b;
    b.vad = {0.0f, 0.0f, 0.0f};
    b.confidence = 0.5f;
    EXPECT_NEAR(smoother.update(b).vad.a, 1.0f - 0.25f, 1e-6f);  // alpha 0.5 x confidence 0.5
}

TEST(LabelHysteresis, IgnoresWobbleButFollowsRealChanges) {
    LabelHysteresis h(0.08f, 0.25f, 2);
    EXPECT_EQ(h.update(prototype(EmotionLabel::Anger)), EmotionLabel::Anger);
    // Midway between anger and fear, tipping slightly toward fear (advantage < margin): no flicker.
    const Vad mid = lerp(prototype(EmotionLabel::Anger), prototype(EmotionLabel::Fear), 0.53f);
    ASSERT_EQ(nearest_label(mid), EmotionLabel::Fear);
    EXPECT_EQ(h.update(mid), EmotionLabel::Anger);
    EXPECT_EQ(h.update(mid), EmotionLabel::Anger);
    // Moderately closer to fear (margin <= advantage < strong margin): two consecutive updates.
    const Vad leaning = lerp(prototype(EmotionLabel::Anger), prototype(EmotionLabel::Fear), 0.58f);
    ASSERT_EQ(nearest_label(leaning), EmotionLabel::Fear);
    EXPECT_EQ(h.update(leaning), EmotionLabel::Anger);
    EXPECT_EQ(h.update(leaning), EmotionLabel::Fear);
    // Clearly sad: switches at once.
    EXPECT_EQ(h.update(prototype(EmotionLabel::Sadness)), EmotionLabel::Sadness);
}

TEST(Emphasis, PicksTheWordWithTheStandoutEnergyPeak) {
    const std::vector<Word> words = {
        {"I", 0.00f, 0.10f, 1.0f}, {"can't", 0.12f, 0.40f, 1.0f}, {"believe", 0.42f, 0.80f, 1.0f},
        {"you", 0.82f, 0.95f, 1.0f}, {"did", 0.97f, 1.15f, 1.0f}, {"this!", 1.17f, 1.50f, 1.0f}};
    std::vector<float> envelope(160, -30.0f);
    for (std::size_t i = 0; i < envelope.size(); ++i) envelope[i] = -24.0f + static_cast<float>(i % 7) * 0.3f;
    for (std::size_t i = 50; i < 70; ++i) envelope[i] = -14.0f;  // "believe" is 10 dB hotter
    const auto emphasis = find_emphasis(words, envelope, 0.01f);
    ASSERT_EQ(emphasis.size(), 1u);
    EXPECT_EQ(emphasis[0], 2u);

    std::vector<float> flat(160, -24.0f);
    EXPECT_TRUE(find_emphasis(words, flat, 0.01f).empty());
}

}  // namespace
}  // namespace ee
