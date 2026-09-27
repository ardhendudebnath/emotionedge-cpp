#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

#include "core/asr/asr_stage.hpp"
#include "core/asr/local_agreement.hpp"
#include "core/asr/scripted_engine.hpp"
#include "support/signals.hpp"
#include "support/test_context.hpp"

namespace ee {
namespace {

std::vector<Word> words(std::initializer_list<const char*> texts) {
    std::vector<Word> out;
    float t = 0.0f;
    for (const char* text : texts) {
        out.push_back({text, t, t + 0.2f, 1.0f});
        t += 0.25f;
    }
    return out;
}

std::vector<std::string> texts(const std::vector<Word>& ws) {
    std::vector<std::string> out;
    for (const Word& w : ws) out.push_back(w.text);
    return out;
}

TEST(LocalAgreement, CommitsWhatTwoConsecutiveHypothesesAgreeOn) {
    LocalAgreement la(2);
    EXPECT_EQ(la.update(words({"I", "can't"})), 0u);  // a single hypothesis is never trusted
    EXPECT_TRUE(la.committed().empty());
    EXPECT_EQ(la.update(words({"I", "can't", "believe"})), 2u);
    EXPECT_EQ(texts(la.committed()), (std::vector<std::string>{"I", "can't"}));
    EXPECT_EQ(texts(la.tentative()), (std::vector<std::string>{"believe"}));
    EXPECT_EQ(la.update(words({"I", "can't", "believe", "you"})), 1u);
    EXPECT_EQ(texts(la.current()), (std::vector<std::string>{"I", "can't", "believe", "you"}));
}

TEST(LocalAgreement, CommittedWordsSurviveRevisions) {
    LocalAgreement la(2);
    la.update(words({"I", "can't", "leave"}));
    la.update(words({"I", "can't", "believe"}));  // agree on "I can't" only
    EXPECT_EQ(texts(la.committed()), (std::vector<std::string>{"I", "can't"}));
    // The engine now revises an early word; the committed prefix stays and alignment falls back to time.
    la.update(words({"Hi", "can't", "believe", "you"}));
    EXPECT_EQ(texts(la.committed()).front(), "I");
    const auto final = la.finalize(words({"I", "can't", "believe", "you", "did", "this!"}));
    EXPECT_EQ(texts(final), (std::vector<std::string>{"I", "can't", "believe", "you", "did", "this!"}));
}

TEST(LocalAgreement, NormalizesCaseAndPunctuation) {
    EXPECT_EQ(LocalAgreement::normalize("This!"), "this");
    EXPECT_EQ(LocalAgreement::normalize("\"can't,\""), "can't");
    LocalAgreement la(2);
    la.update(words({"hello", "world"}));
    EXPECT_EQ(la.update(words({"Hello,", "World."})), 2u);
}

TEST(ScriptedAsrEngine, RevealsWordsAsTheAudioArrives) {
    auto engine = ScriptedAsrEngine::from_json(R"({"language": "en", "words": [
        {"w": "I", "t0": 1.00, "t1": 1.20}, {"w": "can't", "t0": 1.25, "t1": 1.60},
        {"w": "believe", "t0": 1.65, "t1": 2.10}, {"w": "later", "t0": 9.0, "t1": 9.4}]})");
    const std::vector<float> audio(static_cast<std::size_t>(16000 * 0.75), 0.0f);  // 0.9 .. 1.65 s
    AsrRequest req;
    req.audio = audio;
    req.utterance_start_s = 0.9;
    AsrResult partial = engine.transcribe(req);
    ASSERT_EQ(partial.words.size(), 2u);  // "believe" has not ended yet
    EXPECT_NEAR(partial.words[0].t0, 0.10f, 1e-5f);  // relative to the utterance start
    req.final = true;
    const std::vector<float> longer(static_cast<std::size_t>(16000 * 1.4), 0.0f);
    req.audio = longer;
    EXPECT_EQ(engine.transcribe(req).words.size(), 3u);
    EXPECT_EQ(engine.transcribe(req).language, "en");
    EXPECT_THROW((void)ScriptedAsrEngine::from_json(R"({"words": [{"t0": 1}]})"), ConfigError);
}

TEST(StreamingAsrStage, EmitsStablePartialsThenTheFinalTranscript) {
    const auto script_path = std::filesystem::temp_directory_path() / "ee_asr_script.json";
    std::vector<Word> spoken;
    const char* names[] = {"one", "two", "three", "four", "five", "six", "seven", "eight"};
    for (int i = 0; i < 8; ++i) spoken.push_back({names[i], 0.5f + 0.3f * static_cast<float>(i), 0.75f + 0.3f * static_cast<float>(i), 1.0f});
    std::ofstream(script_path) << make_asr_script(spoken, 0.0, "en");

    test::RecordingContext ctx;
    ctx.mutable_pipeline().sample_rate = 16000;
    ctx.mutable_params().set("engine", "scripted");
    ctx.mutable_params().set("script", script_path.string());
    ctx.telemetry().begin_utterance(1);
    StreamingAsrStage stage;
    stage.open(ctx);

    const auto audio = test::silence(3.0, 16000);  // content does not matter to the scripted engine
    for (std::size_t pos = 0; pos < audio.size(); pos += 512) {
        Frame f;
        f.reset(FrameKind::Audio);
        f.utterance = 1;
        f.stream_pos = static_cast<std::int64_t>(pos) + 4000;  // utterance starts at 0.25 s
        f.sample_rate = 16000;
        f.flags = pos == 0 ? frame_flags::kSpeechStart : 0u;
        f.audio.assign(audio.begin() + static_cast<std::ptrdiff_t>(pos),
                       audio.begin() + static_cast<std::ptrdiff_t>(std::min(audio.size(), pos + 512)));
        stage.process(f);
    }
    Frame endpoint;
    endpoint.reset(FrameKind::Audio);
    endpoint.utterance = 1;
    endpoint.flags = frame_flags::kEndpoint | frame_flags::kFinal;
    endpoint.src_start = 0.25;
    endpoint.src_end = 2.9;
    stage.process(endpoint);

    const auto transcripts = ctx.of(FrameKind::Transcript);
    ASSERT_GE(transcripts.size(), 3u);
    std::uint32_t last_stable = 0;
    for (std::size_t i = 0; i + 1 < transcripts.size(); ++i) {
        EXPECT_FALSE(transcripts[i].is_final());
        EXPECT_GE(transcripts[i].stable_words, last_stable);  // committed words only grow
        last_stable = transcripts[i].stable_words;
    }
    const Frame& final = transcripts.back();
    EXPECT_TRUE(final.is_final());
    EXPECT_EQ(final.text, "one two three four five six seven eight");
    EXPECT_NEAR(final.words[0].t0, 0.25f, 1e-4f);  // 0.5 s absolute, utterance starts at 0.25 s
    EXPECT_DOUBLE_EQ(final.src_end, 2.9);
    EXPECT_EQ(final.language, "en");
    EXPECT_GT(ctx.telemetry().rtf().count, 0u);
    EXPECT_GE(ctx.telemetry().milestone_us(1, telemetry::Milestone::AsrFinal), 0);
    std::filesystem::remove(script_path);
}

TEST(StreamingAsrStage, ReportsMissingEnginesClearly) {
    test::RecordingContext ctx;
    ctx.mutable_params().set("engine", "scripted");  // no script
    StreamingAsrStage stage;
    EXPECT_THROW(stage.open(ctx), ConfigError);
#if !defined(EE_HAVE_WHISPER)
    ctx.mutable_params().set("engine", "whisper");
    EXPECT_THROW(stage.open(ctx), ConfigError);
#endif
}

}  // namespace
}  // namespace ee
