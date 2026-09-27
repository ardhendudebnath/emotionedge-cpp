#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>

#include "core/prosody/controller.hpp"
#include "core/prosody/controller_stage.hpp"
#include "core/prosody/expressivity.hpp"
#include "core/prosody/style.hpp"
#include "support/test_context.hpp"

namespace ee {
namespace {

ExpressivityProfile hindi() { return ExpressivityProfiles::defaults().get("hi"); }

TEST(EmotionController, ReproducesTheBlueprintWalkthroughPlan) {
    // Blueprint p.3: anger (V -0.62, A +0.78, D +0.55, conf 0.87) -> Hindi prosody plan
    // "pitch +15%  range +30%  rate +10%  energy +4 dB  pause 80 ms  accent x1.4".
    const EmotionController controller;
    const ProsodyTargets p = controller.plan({-0.62f, 0.78f, 0.55f}, 0.87f, hindi());
    EXPECT_NEAR(p.pitch_pct, 15.0f, 1.0f);
    EXPECT_NEAR(p.range_pct, 30.0f, 1.0f);
    EXPECT_NEAR(p.rate_pct, 10.0f, 0.5f);
    EXPECT_NEAR(p.energy_db, 4.0f, 0.25f);
    EXPECT_NEAR(p.pause_ms, 80.0f, 5.0f);
    EXPECT_NEAR(p.accent, 1.4f, 0.03f);
    EXPECT_GT(p.tension, 0.5f);     // anger: tense voice quality
    EXPECT_GT(p.final_fall, 0.4f);  // dominant: falling final contour
}

TEST(EmotionController, FollowsTheRulesTable) {
    const EmotionController c;
    const auto plan = [&](Vad v) { return c.plan(v, 1.0f, ExpressivityProfiles::defaults().get("en")); };
    // Arousal up: wider range, faster, louder.
    const auto calm = plan({0.0f, -0.5f, 0.0f});
    const auto excited = plan({0.0f, 0.8f, 0.0f});
    EXPECT_GT(excited.range_pct, calm.range_pct);
    EXPECT_GT(excited.rate_pct, calm.rate_pct);
    EXPECT_GT(excited.energy_db, calm.energy_db);
    // Valence down: lower pitch mean when sad, tenser voice when angry.
    EXPECT_LT(plan(prototype(EmotionLabel::Sadness)).pitch_pct, plan({0.0f, -0.5f, -0.35f}).pitch_pct);
    EXPECT_GT(plan(prototype(EmotionLabel::Anger)).tension, plan(prototype(EmotionLabel::Joy)).tension);
    // Dominance up: falling contour, fewer hesitation pauses.
    const auto dominant = plan({0.0f, 0.0f, 0.8f});
    const auto submissive = plan({0.0f, 0.0f, -0.8f});
    EXPECT_GT(dominant.final_fall, submissive.final_fall);
    EXPECT_LT(dominant.hesitation, submissive.hesitation);
    // Emphasis: 60-100 ms pre-pause, accent above 1.
    for (const Vad v : {Vad{0, -1, 0}, Vad{0, 1, -1}, Vad{0, 1, 1}}) {
        const auto p = plan(v);
        EXPECT_GE(p.pause_ms, 60.0f);
        EXPECT_LE(p.pause_ms, 100.0f);
        EXPECT_GT(p.accent, 1.0f);
    }
}

TEST(EmotionController, LowConfidenceShrinksTowardNeutral) {
    const EmotionController c;
    const Vad anger = prototype(EmotionLabel::Anger);
    const auto sure = c.plan(anger, 0.9f, hindi());
    const auto unsure = c.plan(anger, 0.5f, hindi());
    const auto guess = c.plan(anger, 0.2f, hindi());
    EXPECT_GT(sure.range_pct, unsure.range_pct);
    EXPECT_GT(unsure.range_pct, 0.0f);
    EXPECT_FLOAT_EQ(guess.range_pct, 0.0f);
    EXPECT_FLOAT_EQ(guess.pitch_pct, 0.0f);
    EXPECT_FLOAT_EQ(guess.accent, 1.1f);  // emphasis itself is measured, not guessed: it stays
}

TEST(EmotionController, ScalesByTheTargetLanguageProfile) {
    const EmotionController c;
    const auto profiles = ExpressivityProfiles::defaults();
    const Vad anger = prototype(EmotionLabel::Anger);
    EXPECT_LT(c.plan(anger, 0.9f, profiles.get("ja")).range_pct, c.plan(anger, 0.9f, profiles.get("hi")).range_pct);
    // Tonal languages damp pitch range beyond their overall intensity (pitch carries lexical tone).
    const auto zh = c.plan(anger, 0.9f, profiles.get("zh"));
    const auto hi = c.plan(anger, 0.9f, profiles.get("hi"));
    EXPECT_LT(zh.range_pct / hi.range_pct, zh.rate_pct / hi.rate_pct);
    EXPECT_EQ(profiles.get("ja").reg, Register::Formal);
    EXPECT_EQ(profiles.get("xx").language, "xx");  // unknown languages use the default profile
}

TEST(EmotionController, ClosedLoopNudgesAlongMinusDeltaVad) {
    ControllerConfig cfg;
    cfg.loop_gain = 0.5f;
    cfg.max_correction = 0.3f;
    EmotionController c(cfg);
    // The render came out calmer than the source (ΔA = -0.4), measured confidently on arousal only.
    EXPECT_FALSE(c.feedback(0.9f, {0.0f, -0.4f, 0.0f}, {0.0f, 1.0f, 0.0f}));  // ECS fine: no change
    EXPECT_TRUE(c.feedback(0.6f, {-0.5f, -0.4f, 0.2f}, {0.0f, 1.0f, 0.5f}));
    EXPECT_FLOAT_EQ(c.correction().v, 0.0f);   // valence was not measurable: untouched
    EXPECT_FLOAT_EQ(c.correction().a, 0.2f);   // -ΔA · confidence · gain
    EXPECT_FLOAT_EQ(c.correction().d, -0.05f);
    c.feedback(0.5f, {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f, 0.0f});
    EXPECT_FLOAT_EQ(c.correction().a, 0.3f);   // bounded
    const Vad target = c.target({0.0f, 0.9f, 0.0f});
    EXPECT_FLOAT_EQ(target.a, 1.0f);           // clamped to the cube
    c.next_utterance();
    EXPECT_FLOAT_EQ(c.correction().a, 0.15f);  // decays between utterances
}

TEST(StyleBank, BlendsAnchorsContinuously) {
    const StyleBank bank = StyleBank::placeholder();
    const auto cosine = [](const StyleVector& a, const StyleVector& b) {
        double dot = 0, na = 0, nb = 0;
        for (std::size_t i = 0; i < kStyleDim; ++i) {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        return dot / std::sqrt(na * nb);
    };
    const StyleVector at_anger = bank.blend(prototype(EmotionLabel::Anger));
    for (EmotionLabel l : kEmotionLabels) {
        if (l != EmotionLabel::Anger) {
            EXPECT_GT(cosine(at_anger, bank.anchor(EmotionLabel::Anger)), cosine(at_anger, bank.anchor(l)));
        }
    }
    const StyleVector nearby = bank.blend(prototype(EmotionLabel::Anger) + Vad{0.02f, 0.0f, 0.0f});
    EXPECT_GT(cosine(at_anger, nearby), 0.99);  // small emotion change, small style change
    EXPECT_EQ(StyleBank::placeholder().anchor(EmotionLabel::Joy), bank.anchor(EmotionLabel::Joy));  // deterministic
}

TEST(ControllerStage, TurnsTranslationsIntoSpeechRequestsAndAppliesFeedback) {
    test::RecordingContext ctx;
    ctx.telemetry().begin_utterance(1);
    EmotionControllerStage stage;
    stage.open(ctx);
    Frame t;
    t.reset(FrameKind::Translation);
    t.utterance = 1;
    t.flags = frame_flags::kFinal;
    t.text = "मुझे यकीन नहीं हो रहा कि तुमने ऐसा किया!";
    t.language = "hi";
    t.emphasis = {1};
    t.emotion.vad = {-0.62f, 0.78f, 0.55f};
    t.emotion.label = EmotionLabel::Anger;
    t.emotion.confidence = 0.87f;
    stage.process(t);
    const auto speech = ctx.of(FrameKind::Speech);
    ASSERT_EQ(speech.size(), 1u);
    EXPECT_NEAR(speech[0].prosody.pitch_pct, 15.0f, 1.0f);
    EXPECT_EQ(speech[0].emotion.vad, t.emotion.vad);  // the source emotion travels on to 5.2
    EXPECT_EQ(speech[0].emphasis, t.emphasis);
    EXPECT_GE(ctx.telemetry().milestone_us(1, telemetry::Milestone::ControllerDone), 0);

    Frame fb;
    fb.reset(FrameKind::Feedback);
    fb.utterance = 1;
    fb.score = 0.5f;
    fb.delta = {0.0f, -0.4f, 0.0f};
    fb.axis_confidence = {0.0f, 0.9f, 0.0f};
    stage.process(fb);
    EXPECT_GT(stage.controller().correction().a, 0.0f);
}

TEST(Config, ExpressivityFileMatchesBuiltInDefaults) {
    const auto file = ExpressivityProfiles::load(std::string(EE_SOURCE_DIR) + "/config/expressivity.yaml");
    const auto builtin = ExpressivityProfiles::defaults();
    for (const char* lang : {"en", "hi", "es", "it", "fr", "de", "ja", "ko", "zh", "vi", "th"}) {
        SCOPED_TRACE(lang);
        EXPECT_FLOAT_EQ(file.get(lang).intensity, builtin.get(lang).intensity);
        EXPECT_FLOAT_EQ(file.get(lang).pitch_range, builtin.get(lang).pitch_range);
        EXPECT_EQ(file.get(lang).reg, builtin.get(lang).reg);
    }
}

TEST(Config, EmotionSpaceFileMatchesCompiledPrototypes) {
    std::ifstream in(std::string(EE_SOURCE_DIR) + "/config/emotion_space.json");
    ASSERT_TRUE(in.good());
    const auto root = nlohmann::json::parse(in);
    for (EmotionLabel l : kEmotionLabels) {
        SCOPED_TRACE(std::string(to_string(l)));
        const auto& p = root.at("prototypes").at(std::string(to_string(l)));
        EXPECT_FLOAT_EQ(p[0].get<float>(), prototype(l).v);
        EXPECT_FLOAT_EQ(p[1].get<float>(), prototype(l).a);
        EXPECT_FLOAT_EQ(p[2].get<float>(), prototype(l).d);
    }
}

}  // namespace
}  // namespace ee
