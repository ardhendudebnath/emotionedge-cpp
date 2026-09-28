#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <vector>

#include "core/audio/audio_io.hpp"
#include "core/audio/dsp.hpp"
#include "core/audio/pitch.hpp"
#include "core/audio/resampler.hpp"
#include "core/audio/speaker.hpp"
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

// A capped first clause: first audio after a few words, the rest kept whole, emphasis remapped.
TEST(ClauseChunker, CapsTheFirstClauseForSlowTts) {
    ChunkerConfig cfg;
    cfg.max_first_words = 3;
    const auto clauses = chunk_clauses("I really cannot believe you did this to me", "en", {4}, cfg);
    ASSERT_EQ(clauses.size(), 2u);
    EXPECT_EQ(clauses[0].text, "I really cannot");
    EXPECT_EQ(clauses[1].text, "believe you did this to me");
    EXPECT_EQ(clauses[1].emphasis, (std::vector<std::uint16_t>{1}));  // "you"
    // Never leaves fewer than two words behind.
    EXPECT_EQ(chunk_clauses("one two three four", "en", {}, cfg).size(), 1u);
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
    print[0] = 0.120f;  // 120 Hz speaker (pitch-encoder layout)
    SynthesisRequest req;
    req.text = "hello there my friend";
    req.voice = &print;
    req.voice_f0 = voice_print_f0(print);  // what the speaker stage sends with every print
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

TEST(PacingConfig, RampsTheSpeakingRateWithTheBacklog) {
    EXPECT_EQ(PacingConfig{}.speed(10.0), 1.0f);  // off by default
    const PacingConfig pacing{.start_s = 0.25f, .full_s = 1.5f, .max_speed = 1.3f};
    EXPECT_EQ(pacing.speed(0.0), 1.0f);
    EXPECT_EQ(pacing.speed(0.25), 1.0f);
    EXPECT_NEAR(pacing.speed(0.875), 1.15f, 1e-5f);
    EXPECT_NEAR(pacing.speed(9.0), 1.3f, 1e-6f);
    EXPECT_EQ(pacing.speed(0.0, 6.0, 1.0, 0.5), 1.0f);  // no fit term unless fit_next
}

// A translation that would still be playing when the next one is due (the speaker's usual gap
// plus a similar utterance later) builds the queue: it is sped up before any backlog exists.
TEST(PacingConfig, FitsATranslationBeforeTheNextOneIsDue) {
    const PacingConfig pacing{.start_s = 0.25f, .full_s = 1.5f, .max_speed = 1.3f, .fit_next = true};
    EXPECT_EQ(pacing.speed(0.0, 3.0, 2.0, 1.0), 1.0f);             // 3 s fits in 2 s + a 1 s gap
    EXPECT_NEAR(pacing.speed(0.0, 3.6, 2.0, 1.0), 1.2f, 1e-5f);    // 3.6 s does not: 1.2x
    EXPECT_EQ(pacing.speed(0.0, 3.6, 2.0, 3.0), 1.0f);             // a speaker who pauses leaves room
    EXPECT_NEAR(pacing.speed(0.0, 10.0, 2.0, 1.0), 1.3f, 1e-6f);   // capped
    EXPECT_NEAR(pacing.speed(1.5, 2.0, 2.0, 1.0), 1.3f, 1e-6f);    // the backlog term still applies
}

// Samples per clause of one utterance rendered by a TTS stage with pacing up to 1.3x.
struct PacedRender {
    AudioIo io;
    test::RecordingContext ctx;
    TtsStage stage;

    PacedRender() {
        ctx.services().audio = &io;
        ctx.mutable_params().set("calibration", "false");
        ctx.mutable_params().set("pacing.max_speed", "1.3");
        stage.open(ctx);
    }
    void say(std::uint64_t utterance, const std::string& text) {
        Frame speech;
        speech.reset(FrameKind::Speech);
        speech.utterance = utterance;
        speech.language = "en";
        speech.text = text;
        stage.process(speech);
    }
    std::vector<std::size_t> clauses(std::uint64_t utterance) const {
        std::vector<std::size_t> out{0};
        for (const Frame& f : ctx.of(FrameKind::SynthAudio)) {
            if (f.utterance != utterance) continue;
            out.back() += f.audio.size();
            if (f.has(frame_flags::kClauseEnd)) out.push_back(0);
        }
        out.pop_back();
        return out;
    }
};

// A translation that would wait behind the previous one is spoken faster, at one rate for the
// whole utterance, even once the queue has drained.
TEST(TtsStage, PacesAnUtteranceByThePlayoutBacklog) {
    const std::string text = "First clause here, and a second clause, and then a third clause.";
    PacedRender normal;
    normal.say(1, text);
    normal.stage.close();
    PacedRender behind;
    behind.io.playout_queued_s.store(3.0);
    behind.say(1, text);
    behind.io.playout_queued_s.store(0.0);
    behind.stage.close();

    const auto a = normal.clauses(1);
    const auto b = behind.clauses(1);
    ASSERT_GE(a.size(), 2u);
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double ratio = static_cast<double>(b[i]) / static_cast<double>(a[i]);
        EXPECT_LT(ratio, 0.9) << "clause " << i;  // syllables at 1/1.3; the formant voice's word gaps stay
        EXPECT_GT(ratio, 1.0 / 1.3 - 0.02) << "clause " << i;
    }
}

// Clauses still waiting in the TTS are backlog too: they have not reached playback yet.
TEST(TtsStage, CountsItsOwnQueuedClausesAsBacklog) {
    const std::string next = "And now the next sentence.";
    PacedRender alone;
    alone.say(2, next);
    PacedRender queued;
    queued.say(1, "A long first clause with many words in it, a second clause just as long as the first, "
                  "and a third one that is longer still before it ends.");
    queued.say(2, next);  // the first sentence's later clauses are still waiting
    queued.stage.close();
    ASSERT_EQ(alone.clauses(2).size(), 1u);
    ASSERT_EQ(queued.clauses(2).size(), 1u);
    EXPECT_LT(queued.clauses(2)[0], alone.clauses(2)[0] * 9 / 10);
}

}  // namespace
}  // namespace ee
