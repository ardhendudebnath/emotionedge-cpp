#include <gtest/gtest.h>

#include <cstdlib>

#include "core/translate/control_tokens.hpp"
#include "core/translate/emphasis.hpp"
#include "core/translate/glossary.hpp"
#include "core/translate/languages.hpp"
#include "core/translate/phrasebook.hpp"
#include "core/translate/translate_stage.hpp"
#include "core/translate/wait_k.hpp"
#include "support/test_context.hpp"

namespace ee {
namespace {

const std::string kHindi = "मुझे यकीन नहीं हो रहा कि तुमने ऐसा किया!";

EmotionState walkthrough_emotion() {
    // Blueprint p.3, "2 · FELT": anger, V -0.62, A +0.78, D +0.55, confidence 0.87.
    EmotionState s;
    s.vad = {-0.62f, 0.78f, 0.55f};
    s.label = EmotionLabel::Anger;
    s.confidence = 0.87f;
    return s;
}

TEST(ControlTokens, MatchTheBlueprintFormat) {
    const ControlTokens t = make_control_tokens(walkthrough_emotion(), Register::Casual, 0.35f);
    EXPECT_EQ(format_control_prefix(t), "<emo=anger a=0.78 reg=casual>");          // walkthrough (p.3)
    EXPECT_EQ(format_control_prefix(make_control_tokens(walkthrough_emotion(), Register::Casual, 0.35f, 0.1f), 0.1f),
              "<emo=anger a=0.8 reg=casual>");                                     // system diagram (3.2)
    EmotionState unsure = walkthrough_emotion();
    unsure.confidence = 0.2f;  // below τ: neutral fallback
    EXPECT_EQ(format_control_prefix(make_control_tokens(unsure, Register::Formal, 0.35f)), "<emo=neutral a=0.00 reg=formal>");
    EXPECT_EQ(strip_control_prefix("<emo=anger a=0.78 reg=casual> I can't"), "I can't");
    EXPECT_EQ(strip_control_prefix("no tokens here"), "no tokens here");
}

// An echoed prefix (vanilla NLLB, then kept by a wait-k draft) must never reach TTS or captions.
TEST(ControlTokens, EchoedTokensAreRemovedFromOutput) {
    EXPECT_EQ(remove_control_tokens("<emo=surprise a=0.4 reg=casual> \xE0\xA4\xB9\xE0\xA4\xBE\xE0\xA4\x81"),
              "\xE0\xA4\xB9\xE0\xA4\xBE\xE0\xA4\x81");
    EXPECT_EQ(remove_control_tokens("one <emo=joy a=0.5 reg=casual> two <emo=anger a=0.8 reg=formal>"), "one two");
    EXPECT_EQ(remove_control_tokens("no tokens, <b>tags</b> stay"), "no tokens, <b>tags</b> stay");
    EXPECT_EQ(remove_control_tokens("unterminated <emo=joy a=0.5"), "unterminated <emo=joy a=0.5");
}

TEST(ControlTokens, EmphasisMarkupRoundTrips) {
    const std::vector<std::string> source = {"I", "can't", "believe", "you", "did", "this!"};
    const std::string marked = apply_emphasis_markup(source, {2});
    EXPECT_EQ(marked, "I can't <em>believe</em> you did this!");
    const Markup parsed = parse_markup(marked);
    EXPECT_EQ(parsed.plain, "I can't believe you did this!");
    EXPECT_EQ(parsed.emphasis, (std::vector<std::uint16_t>{2}));

    const Markup hindi = parse_markup("मुझे <em>यकीन</em> नहीं हो रहा कि तुमने ऐसा किया!", "hi");
    EXPECT_EQ(hindi.plain, kHindi);
    EXPECT_EQ(hindi.emphasis, (std::vector<std::uint16_t>{1}));
}

TEST(Glossary, LocksTermsThroughTranslation) {
    Glossary g;
    g.add("ONNX Runtime");
    g.add("Wi-Fi", "वाई-फ़ाई");
    const auto p = g.protect("Does onnx runtime need Wi-Fi? Not ONNXRuntime.");
    EXPECT_EQ(p.text, "Does __T0__ need __T1__? Not ONNXRuntime.");  // whole words only
    std::uint32_t missing = 0;
    EXPECT_EQ(g.restore("क्या __T0__ को __T1__ चाहिए?", p, &missing), "क्या onnx runtime को वाई-फ़ाई चाहिए?");
    EXPECT_EQ(missing, 0u);
    (void)g.restore("__T1__ only", p, &missing);
    EXPECT_EQ(missing, 1u);
}

TEST(WaitK, BudgetsAndKeepsTheCommittedPrefix) {
    const WaitKPolicy policy(3);
    EXPECT_EQ(policy.target_budget(2), 0u);
    EXPECT_EQ(policy.target_budget(3), 1u);
    EXPECT_EQ(policy.target_budget(6), 4u);
    std::vector<std::string> committed;
    EXPECT_EQ(policy.extend(committed, {"a", "b", "c", "d"}, 4), 2u);
    EXPECT_EQ(committed, (std::vector<std::string>{"a", "b"}));
    // A later draft disagrees about "b": the committed prefix is kept, new words are appended.
    EXPECT_EQ(policy.extend(committed, {"a", "X", "c", "d", "e"}, 5), 1u);
    EXPECT_EQ(committed, (std::vector<std::string>{"a", "b", "c"}));
    EXPECT_EQ(WaitKPolicy(2, 1.5f).target_budget(4), 4u);
}

TEST(Emphasis, PrefersMarkupThenAlignmentThenPosition) {
    EXPECT_EQ(project_emphasis({2}, 6, 8, {1}, {}), (std::vector<std::uint16_t>{1}));
    EXPECT_EQ(project_emphasis({2}, 6, 8, {}, {{2, 5}, {3, 6}}), (std::vector<std::uint16_t>{5}));
    EXPECT_EQ(project_emphasis({5}, 6, 11, {}, {}), (std::vector<std::uint16_t>{10}));  // last -> last
    EXPECT_TRUE(project_emphasis({}, 6, 8, {}, {}).empty());
}

TEST(Languages, CodesAndWordSplitting) {
    EXPECT_EQ(nllb_code("hi").value(), "hin_Deva");
    EXPECT_EQ(nllb_code("en").value(), "eng_Latn");
    EXPECT_FALSE(nllb_code("xx").has_value());
    EXPECT_EQ(split_words(kHindi, "hi").size(), 9u);
    EXPECT_EQ(split_words("你好世界", "zh"), (std::vector<std::string>{"你", "好", "世", "界"}));
    EXPECT_EQ(join_text({"你", "好"}, "zh"), "你好");
}

TEST(Phrasebook, TranslatesTheWalkthroughAndCarriesEmphasis) {
    PhrasebookTranslator book = PhrasebookTranslator::load(std::string(EE_SOURCE_DIR) + "/config/phrasebook.yaml");
    TranslationRequest req;
    req.source = "<emo=anger a=0.78 reg=casual> I can\xE2\x80\x99t <em>believe</em> you did this!";
    req.source_language = "en";
    req.target_language = "hi";
    const TranslationResult r = book.translate(req);
    EXPECT_EQ(r.text, "मुझे <em>यकीन</em> नहीं हो रहा कि तुमने ऐसा किया!");

    req.source = "Something the phrasebook has never seen";
    EXPECT_EQ(book.translate(req).text, "[hi] Something the phrasebook has never seen");
}

TEST(TranslateStage, EmitsTheBlueprintWalkthrough) {
    test::RecordingContext ctx;
    ctx.mutable_pipeline().source_language = "en";
    ctx.mutable_pipeline().target_language = "hi";
    ctx.mutable_params().set("engine", "phrasebook");
    ctx.mutable_params().set("phrasebook", std::string(EE_SOURCE_DIR) + "/config/phrasebook.yaml");
    ctx.telemetry().begin_utterance(1);
    TranslateStage stage;
    stage.open(ctx);

    Frame u;
    u.reset(FrameKind::Utterance);
    u.utterance = 1;
    u.flags = frame_flags::kFinal;
    u.language = "en";
    u.text = "I can't believe you did this!";
    const char* ws[] = {"I", "can't", "believe", "you", "did", "this!"};
    for (const char* w : ws) u.words.push_back({w, 0.0f, 0.0f, 1.0f});
    u.emphasis = {2};
    u.emotion = walkthrough_emotion();
    stage.process(u);

    const auto out = ctx.of(FrameKind::Translation);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_TRUE(out[0].is_final());
    // Blueprint p.3 "3 · MT INPUT" and "4 · TRANSLATED".
    EXPECT_EQ(out[0].detail, "<emo=anger a=0.78 reg=casual> I can't <em>believe</em> you did this!");
    EXPECT_EQ(out[0].text, kHindi);
    EXPECT_EQ(out[0].emphasis, (std::vector<std::uint16_t>{1}));  // "यकीन"
    EXPECT_EQ(out[0].emotion.label, EmotionLabel::Anger);
    EXPECT_GE(ctx.telemetry().milestone_us(1, telemetry::Milestone::MtFinal), 0);
}

// Streaming: the leading words go ahead, then the whole translation, which starts with them.
// An emphasized utterance needs the whole alignment, so it is translated in one pass.
TEST(TranslateStage, StreamsTheLeadingWordsAhead) {
    test::RecordingContext ctx;
    ctx.mutable_pipeline().source_language = "en";
    ctx.mutable_pipeline().target_language = "hi";
    ctx.mutable_params().set("engine", "phrasebook");
    ctx.mutable_params().set("phrasebook", std::string(EE_SOURCE_DIR) + "/config/phrasebook.yaml");
    ctx.mutable_params().set("stream", "on");
    ctx.mutable_params().set("stream_words", "3");
    ctx.telemetry().begin_utterance(1);
    TranslateStage stage;
    stage.open(ctx);

    Frame u;
    u.reset(FrameKind::Utterance);
    u.utterance = 1;
    u.flags = frame_flags::kFinal;
    u.language = "en";
    u.text = "I can't believe you did this!";
    for (const char* w : {"I", "can't", "believe", "you", "did", "this!"}) u.words.push_back({w, 0.0f, 0.0f, 1.0f});
    u.emotion = walkthrough_emotion();
    stage.process(u);

    const auto out = ctx.of(FrameKind::Translation);
    ASSERT_EQ(out.size(), 2u);
    EXPECT_TRUE(out[0].has(frame_flags::kLeading));
    EXPECT_FALSE(out[0].is_final());
    EXPECT_EQ(out[0].text, "मुझे यकीन नहीं");
    EXPECT_GE(out[0].expected_words, 4u);  // more than the words sent ahead
    EXPECT_EQ(out[0].emotion.label, EmotionLabel::Anger);
    EXPECT_TRUE(out[1].is_final());
    EXPECT_EQ(out[1].text, kHindi);
    EXPECT_GE(ctx.telemetry().milestone_us(1, telemetry::Milestone::MtFinal), 0);

    u.utterance = 2;
    u.emphasis = {2};
    stage.process(u);
    const auto all = ctx.of(FrameKind::Translation);
    ASSERT_EQ(all.size(), 3u);
    EXPECT_TRUE(all[2].is_final());
    EXPECT_EQ(all[2].emphasis, (std::vector<std::uint16_t>{1}));
}

#if defined(EE_HAVE_CTRANSLATE2)
// The streaming final pass on the real NLLB, when it is present: EE_NLLB_MODEL_DIR=<CTranslate2
// export with sentencepiece.bpe.model>. The leading words are the greedy translation's first
// words, after a target prefix too (CTranslate2's callback reports the prefix's positions as
// well), and the rest after them gives the same whole.
TEST(Ct2Translator, StopsAfterTheLeadingWords) {
    const char* dir = std::getenv("EE_NLLB_MODEL_DIR");
    if (dir == nullptr) GTEST_SKIP() << "set EE_NLLB_MODEL_DIR to run";
    Params p;
    p.set("device", "cpu");
    p.set("stream", "on");
    p.set("beam", "1");
    p.set("control_tokens", "on");
    p.set("warmup", "false");
    const auto mt = make_ct2_translator(dir, p);
    ASSERT_TRUE(mt->streams());
    TranslationRequest r;
    r.source = "<emo=neutral a=0.0 reg=casual> And so, my fellow Americans, ask not what your country can do for you.";
    r.source_language = "en";
    r.target_language = "hi";
    const std::string whole = mt->translate(r).text;
    const std::vector<std::string> words = split_words(whole, "hi");
    ASSERT_GT(words.size(), 4u);

    r.stop_after_words = 2;
    const TranslationResult lead = mt->translate(r);
    EXPECT_TRUE(lead.partial);
    EXPECT_EQ(lead.text, join_text({words[0], words[1]}, "hi"));
    r.target_prefix = {words[0], words[1]};
    const TranslationResult more = mt->translate(r);
    EXPECT_TRUE(more.partial);
    EXPECT_EQ(more.text, join_text({words[0], words[1], words[2], words[3]}, "hi"));
    r.stop_after_words = 0;
    EXPECT_EQ(mt->translate(r).text, whole);

    // An emphasized word's alignment comes with the leading words, as the whole has it.
    TranslationRequest e;
    e.source = "<emo=neutral a=0.0 reg=casual> I can't <em>believe</em> you did this.";
    e.source_language = "en";
    e.target_language = "hi";
    e.want_alignment = true;
    const TranslationResult all = mt->translate(e);
    ASSERT_EQ(all.alignment.size(), 1u);
    ASSERT_LT(all.alignment[0].second + 2u, split_words(all.text, "hi").size());
    e.stop_after_words = all.alignment[0].second + 1u;  // up to the emphasized word
    const TranslationResult head = mt->translate(e);
    EXPECT_TRUE(head.partial);
    EXPECT_EQ(head.alignment, all.alignment);
}
#endif

TEST(TranslateStage, DraftsFollowWaitK) {
    test::RecordingContext ctx;
    ctx.mutable_pipeline().target_language = "hi";
    ctx.mutable_params().set("wait_k", "2");
    TranslateStage stage;
    stage.open(ctx);  // no phrasebook: pseudo-translations make the budget visible
    Frame t;
    t.reset(FrameKind::Transcript);
    t.utterance = 4;
    t.language = "en";
    for (const char* w : {"we", "should", "leave", "now"}) t.words.push_back({w, 0.0f, 0.0f, 1.0f});
    t.stable_words = 1;
    stage.process(t);
    EXPECT_TRUE(ctx.of(FrameKind::Translation).empty());  // wait-2: nothing after one word
    t.stable_words = 3;
    stage.process(t);
    const auto drafts = ctx.of(FrameKind::Translation);
    ASSERT_EQ(drafts.size(), 1u);
    EXPECT_FALSE(drafts[0].is_final());
    EXPECT_EQ(drafts[0].stable_words, 2u);  // (3 - 2 + 1) target words
}

}  // namespace
}  // namespace ee
