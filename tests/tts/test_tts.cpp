#include <gtest/gtest.h>

#include <algorithm>

#include "core/audio/dsp.hpp"
#include "core/audio/pitch.hpp"
#include "core/audio/resampler.hpp"
#include "core/tts/clause_chunker.hpp"
#include "core/tts/formant_synth.hpp"
#include "core/tts/tts_stage.hpp"
#include "support/test_context.hpp"

namespace ee {
namespace {

TEST(ClauseChunker, SplitsAtClausePunctuationKeepingAShortFirstClause) {
    const auto clauses = chunk_clauses("Well, I looked everywhere for it; then I found it under the sofa. Done!", "en", {1, 12});
    ASSERT_EQ(clauses.size(), 2u);
    EXPECT_EQ(clauses[0].text, "Well, I looked everywhere for it;");  // "Well," alone is under min_first_chars
    EXPECT_EQ(clauses[1].text, "then I found it under the sofa. Done!");  // the short tail is merged
    EXPECT_EQ(clauses[0].emphasis, (std::vector<std::uint16_t>{1}));
    EXPECT_EQ(clauses[1].first_word, 6u);
    EXPECT_EQ(clauses[1].emphasis, (std::vector<std::uint16_t>{6}));  // "sofa." is word 12 overall
}

TEST(ClauseChunker, HandlesDevanagariAndRunOnText) {
    const auto hindi = chunk_clauses("मुझे बहुत अफ़सोस है। मेरा इरादा तुम्हें दुख पहुँचाने का नहीं था।", "hi", {2});
    ASSERT_EQ(hindi.size(), 2u);
    EXPECT_EQ(hindi[0].text, "मुझे बहुत अफ़सोस है।");
    EXPECT_EQ(hindi[0].emphasis, (std::vector<std::uint16_t>{2}));

    std::string run_on;
    for (int i = 0; i < 60; ++i) run_on += "word ";
    ChunkerConfig cfg;
    cfg.max_chars = 60;
    for (const Clause& c : chunk_clauses(run_on, "en", {}, cfg)) EXPECT_LE(c.text.size(), 64u);
    EXPECT_TRUE(ends_clause("this!"));
    EXPECT_TRUE(ends_clause("है।"));
    EXPECT_TRUE(ends_clause("said,\""));
    EXPECT_FALSE(ends_clause("hello"));
}

TEST(FormantSynth, CountsSyllables) {
    EXPECT_EQ(count_syllables("I"), 1);
    EXPECT_EQ(count_syllables("believe"), 2);
    EXPECT_EQ(count_syllables("amazing,"), 3);
    EXPECT_EQ(count_syllables("the"), 1);
    EXPECT_EQ(count_syllables("यकीन"), 3);
    EXPECT_EQ(count_syllables("तुम्हें"), 2);  // the virama joins म्ह into one syllable
}

struct Rendered {
    SynthesisResult result;
    double seconds = 0.0;
    float level_db = 0.0f;
    float median_f0 = 0.0f;
};

Rendered render(const ProsodyTargets& p, std::vector<std::uint16_t> emphasis = {}, bool final = true) {
    FormantSynth synth;
    SynthesisRequest req;
    req.text = "I can't believe you did this!";
    req.language = "en";
    req.prosody = p;
    req.emphasis = std::move(emphasis);
    req.utterance_final = final;
    Rendered r;
    synth.synthesize(req, r.result);
    r.seconds = static_cast<double>(r.result.audio.size()) / r.result.sample_rate;
    std::vector<float> active;
    for (float s : r.result.audio) {
        if (std::abs(s) > 1e-3f) active.push_back(s);
    }
    r.level_db = rms_dbfs(active);
    const auto at16k = Resampler::convert(r.result.audio, r.result.sample_rate, 16000);
    std::vector<float> f0;
    for (const PitchFrame& pf : track_pitch(at16k, PitchConfig{})) {
        if (pf.voiced) f0.push_back(pf.f0_hz);
    }
    std::sort(f0.begin(), f0.end());
    r.median_f0 = f0.empty() ? 0.0f : f0[f0.size() / 2];
    return r;
}

TEST(FormantSynth, RendersTheProsodyPlan) {
    const Rendered neutral = render({});
    ProsodyTargets fast;
    fast.rate_pct = 50.0f;
    EXPECT_LT(render(fast).seconds, neutral.seconds * 0.8);
    ProsodyTargets loud;
    loud.energy_db = 6.0f;
    EXPECT_NEAR(render(loud).level_db - neutral.level_db, 6.0f, 1.0f);
    ProsodyTargets high;
    high.pitch_pct = 20.0f;
    EXPECT_NEAR(render(high).median_f0 / neutral.median_f0, 1.2f, 0.06f);
}

TEST(FormantSynth, EmphasisAddsAPrePauseAndLoudness) {
    ProsodyTargets p;
    p.pause_ms = 90.0f;
    p.accent = 1.4f;
    const Rendered plain = render(p);
    const Rendered stressed = render(p, {2});
    const auto& w0 = plain.result.words;
    const auto& w1 = stressed.result.words;
    ASSERT_EQ(w0.size(), 6u);
    ASSERT_EQ(w1.size(), 6u);
    // "believe" starts later by the pre-pause, and every word keeps increasing timestamps.
    EXPECT_NEAR((w1[2].t0 - w1[1].t1) - (w0[2].t0 - w0[1].t1), 0.09f, 0.005f);
    for (std::size_t i = 1; i < w1.size(); ++i) EXPECT_GT(w1[i].t0, w1[i - 1].t1);
}

TEST(FormantSynth, KeepsTheSpeakersPitchFromTheVoicePrint) {
    FormantSynth synth;
    SpeakerEmbedding print{};
    print[0] = 0.120f;  // 120 Hz speaker
    SynthesisRequest req;
    req.text = "hello there my friend";
    req.voice = &print;
    SynthesisResult out;
    synth.synthesize(req, out);
    const auto at16k = Resampler::convert(out.audio, out.sample_rate, 16000);
    std::vector<float> f0;
    for (const PitchFrame& pf : track_pitch(at16k, PitchConfig{})) {
        if (pf.voiced) f0.push_back(pf.f0_hz);
    }
    std::sort(f0.begin(), f0.end());
    ASSERT_FALSE(f0.empty());
    EXPECT_NEAR(f0[f0.size() / 2], 120.0f, 20.0f);
}

TEST(TtsStage, StreamsClausesAndPublishesACalibrationRender) {
    test::RecordingContext ctx;
    ctx.mutable_pipeline().target_language = "hi";
    ctx.telemetry().begin_utterance(3);
    TtsStage stage;
    stage.open(ctx);
    ASSERT_EQ(ctx.emitted.size(), 1u);
    EXPECT_TRUE(ctx.emitted[0].has(frame_flags::kCalibration));

    Frame speech;
    speech.reset(FrameKind::Speech);
    speech.utterance = 3;
    speech.text = "मुझे बहुत अफ़सोस है, मेरा इरादा तुम्हें दुख पहुँचाने का नहीं था।";
    speech.language = "hi";
    speech.emphasis = {2};
    speech.emotion.vad = prototype(EmotionLabel::Sadness);
    stage.process(speech);  // first clause immediately
    const std::size_t after_first = ctx.of(FrameKind::SynthAudio).size();
    EXPECT_GE(ctx.telemetry().milestone_us(3, telemetry::Milestone::TtsFirstChunk), 0);
    stage.tick();  // second clause on the next tick
    const auto chunks = ctx.of(FrameKind::SynthAudio);
    EXPECT_GT(chunks.size(), after_first);

    int clause_ends = 0;
    for (std::size_t i = 1; i < chunks.size(); ++i) {
        const Frame& c = chunks[i];
        EXPECT_EQ(c.utterance, 3u);
        EXPECT_EQ(c.sample_rate, 24000);
        EXPECT_LE(c.audio.size(), 2400u);  // 100 ms chunks
        EXPECT_EQ(c.emotion.vad, speech.emotion.vad);
        clause_ends += c.has(frame_flags::kClauseEnd) ? 1 : 0;
    }
    EXPECT_EQ(clause_ends, 2);
    EXPECT_TRUE(chunks.back().is_final());
}

TEST(TtsStage, BargeInDropsQueuedClauses) {
    test::RecordingContext ctx;
    ctx.mutable_params().set("calibration", "false");
    TtsStage stage;
    stage.open(ctx);
    Frame speech;
    speech.reset(FrameKind::Speech);
    speech.utterance = 1;
    speech.language = "en";
    speech.text = "First clause here, and a second clause, and then a third clause.";
    stage.process(speech);
    const std::size_t first = ctx.emitted.size();
    Frame barge;
    barge.reset(FrameKind::Control);
    barge.flags = frame_flags::kBargeIn;
    barge.utterance = 2;
    stage.process(barge);
    stage.close();  // would synthesize anything still queued
    EXPECT_EQ(ctx.emitted.size(), first);
}

}  // namespace
}  // namespace ee
