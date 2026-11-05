// Phase-2 classifier plumbing: the byte-level BPE tokenizer (DistilRoBERTa) and the class ->
// V·A·D mapping shared by the lexical and acoustic ONNX models.
#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <thread>
#include <vector>

#include "core/emotion/acoustic.hpp"
#include "core/emotion/bpe_tokenizer.hpp"
#include "core/emotion/class_mapping.hpp"
#include "core/emotion/emotion_types.hpp"
#include "core/runtime/params.hpp"

namespace ee {
namespace {

namespace fs = std::filesystem;

// ---- byte-level BPE ------------------------------------------------------------------------

ByteLevelBpeTokenizer tiny_tokenizer() {
    // U+0120 ("\xC4\xA0") is the byte-level symbol for a space.
    std::unordered_map<std::string, std::int64_t> vocab{{"<s>", 0}, {"</s>", 2}, {"hello", 10}, {"\xC4\xA0w", 11},
                                                        {"o", 12},  {"r", 13},   {"l", 14},     {"d", 15},
                                                        {"!", 16},  {"h", 17},   {"e", 18}};
    std::vector<std::pair<std::string, std::string>> merges{{"h", "e"}, {"l", "l"}, {"he", "ll"}, {"hell", "o"},
                                                            {"\xC4\xA0", "w"}};
    return ByteLevelBpeTokenizer(std::move(vocab), merges, 0, 2);
}

TEST(BpeTokenizer, PreTokenizesLikeTheGpt2Pattern) {
    const ByteLevelBpeTokenizer tok = tiny_tokenizer();
    const std::vector<std::string> expected{"Hello", " world", "!",  " I",  "'m",   " fine", ",",
                                            " ",     " ok",    "\n", "\n", "yes", " 123",  " abc"};
    EXPECT_EQ(tok.pre_tokenize("Hello world! I'm fine,  ok\n\nyes 123 abc"), expected);
    // Contractions are case-sensitive; a space joins the following symbol run; trailing space stays.
    EXPECT_EQ(tok.pre_tokenize("DON'T you're 's '?! "),
              (std::vector<std::string>{"DON", "'", "T", " you", "'re", " '", "s", " '?!", " "}));
    // Latin-1 letters stay inside words, curly quotes and emoji are symbols.
    EXPECT_EQ(tok.pre_tokenize("caf\xC3\xA9 \xE2\x80\x9Cno\xE2\x80\x9D \xF0\x9F\x98\x80"),
              (std::vector<std::string>{"caf\xC3\xA9", " \xE2\x80\x9C", "no", "\xE2\x80\x9D", " \xF0\x9F\x98\x80"}));
}

TEST(BpeTokenizer, MergesByRankAndWrapsWithSpecialTokens) {
    const ByteLevelBpeTokenizer tok = tiny_tokenizer();
    EXPECT_EQ(tok.bpe("hello"), (std::vector<std::string>{"hello"}));
    EXPECT_EQ(tok.bpe(" world"), (std::vector<std::string>{"\xC4\xA0w", "o", "r", "l", "d"}));
    EXPECT_EQ(tok.encode("hello world!"), (std::vector<std::int64_t>{0, 10, 11, 12, 13, 14, 15, 16, 2}));
    // Truncation keeps <s> ... </s>.
    EXPECT_EQ(tok.encode("hello world!", 4), (std::vector<std::int64_t>{0, 10, 11, 2}));
}

TEST(BpeTokenizer, ReadsHuggingFaceTokenizerJson) {
    const fs::path path = fs::temp_directory_path() / "ee_test_tokenizer.json";
    nlohmann::json j;
    j["model"] = {{"type", "BPE"},
                  {"vocab", {{"<s>", 0}, {"</s>", 2}, {"h", 3}, {"i", 4}, {"hi", 5}}},
                  {"merges", nlohmann::json::array({"h i"})}};
    j["pre_tokenizer"] = {{"type", "ByteLevel"}, {"add_prefix_space", false}};
    j["post_processor"] = {{"type", "RobertaProcessing"}, {"cls", {"<s>", 0}}, {"sep", {"</s>", 2}}};
    std::ofstream(path) << j.dump();
    const ByteLevelBpeTokenizer tok = ByteLevelBpeTokenizer::from_tokenizer_json(path);
    EXPECT_EQ(tok.encode("hi"), (std::vector<std::int64_t>{0, 5, 2}));
    fs::remove(path);
    EXPECT_THROW((void)ByteLevelBpeTokenizer::from_tokenizer_json(path), ConfigError);
}

// Exact agreement with Hugging Face on the real DistilRoBERTa vocab, when the model is present:
// EE_LEXICAL_MODEL_DIR=<dir with tokenizer.json>. The expected ids come from the Python tokenizer
// (ml/export/export_lexical_onnx.py --golden).
TEST(BpeTokenizer, MatchesHuggingFaceOnTheRealVocab) {
    const char* dir = std::getenv("EE_LEXICAL_MODEL_DIR");
    if (dir == nullptr) GTEST_SKIP() << "set EE_LEXICAL_MODEL_DIR to run";
    const ByteLevelBpeTokenizer tok = ByteLevelBpeTokenizer::from_tokenizer_json(fs::path(dir) / "tokenizer.json");
    std::ifstream in(fs::path(EE_SOURCE_DIR) / "tests" / "golden" / "roberta_tokens.json");
    const nlohmann::json golden = nlohmann::json::parse(in);
    ASSERT_FALSE(golden.empty());
    for (const auto& item : golden) {
        const std::string text = item.at("text").get<std::string>();
        EXPECT_EQ(tok.encode(text), item.at("ids").get<std::vector<std::int64_t>>()) << text;
    }
}

// ---- class -> V·A·D ---------------------------------------------------------------------------

fs::path write_labels(const std::string& name) {  // one file per test: ctest runs tests in parallel
    const fs::path path = fs::temp_directory_path() / ("ee_test_labels_" + name + ".json");
    nlohmann::json j;
    j["labels"] = {"angry", "neutral", "sad", "other"};
    j["vad"] = {{"angry", {-0.62, 0.78, 0.55}}, {"neutral", {0.0, 0.0, 0.0}}, {"sad", {-0.65, -0.5, -0.35}}};
    j["abstain"] = {"other"};
    j["neutral"] = "neutral";
    j["reliability"] = {0.5, 0.9, 0.6};
    std::ofstream(path) << j.dump();
    return path;
}

TEST(ClassEmotionMap, OneHotClassSitsOnItsPositionWithFullConfidence) {
    const ClassEmotionMap map = ClassEmotionMap::from_file(write_labels("onehot"));
    const ModalityEstimate e = map.from_probabilities(std::vector<float>{1.0f, 0.0f, 0.0f, 0.0f});
    ASSERT_TRUE(e.valid);
    EXPECT_NEAR(e.vad.v, -0.62f, 1e-5f);
    EXPECT_NEAR(e.vad.a, 0.78f, 1e-5f);
    EXPECT_NEAR(e.confidence.v, 0.5f, 1e-5f);  // reliability · 1 · 1
    EXPECT_NEAR(e.confidence.a, 0.9f, 1e-5f);
}

TEST(ClassEmotionMap, NeutralTextIsWeakEvidenceAndAbstainMassScalesConfidence) {
    const ClassEmotionMap map = ClassEmotionMap::from_file(write_labels("neutral"));
    const ModalityEstimate neutral = map.from_probabilities(std::vector<float>{0.0f, 1.0f, 0.0f, 0.0f});
    EXPECT_NEAR(neutral.vad.norm(), 0.0f, 1e-6f);
    EXPECT_NEAR(neutral.confidence.a, 0.9f * 0.3f, 1e-5f);

    // Half the mass on "other": the rest renormalizes to a clear "sad", at half confidence.
    const ModalityEstimate sad = map.from_probabilities(std::vector<float>{0.0f, 0.0f, 0.5f, 0.5f});
    EXPECT_NEAR(sad.vad.v, -0.65f, 1e-5f);
    EXPECT_NEAR(sad.confidence.a, 0.9f * 0.5f, 1e-5f);

    // A split between two emotions lands between them with lower certainty.
    const ModalityEstimate mixed = map.from_probabilities(std::vector<float>{0.5f, 0.0f, 0.5f, 0.0f});
    EXPECT_NEAR(mixed.vad.a, 0.14f, 1e-5f);
    EXPECT_LT(mixed.confidence.a, 0.9f);

    // Logits go through a softmax; a wrong class count abstains.
    const ModalityEstimate from_logits = map.from_logits(std::vector<float>{8.0f, 0.0f, 0.0f, 0.0f});
    EXPECT_EQ(nearest_label(from_logits.vad), EmotionLabel::Anger);
    EXPECT_FALSE(map.from_probabilities(std::vector<float>{1.0f}).valid);
}

TEST(ClassEmotionMap, RejectsLabelsWithoutPosition) {
    const fs::path path = fs::temp_directory_path() / "ee_test_labels_bad.json";
    std::ofstream(path) << R"({"labels": ["joy", "fear"], "vad": {"joy": [0.75, 0.5, 0.35]}})";
    EXPECT_THROW((void)ClassEmotionMap::from_file(path), ConfigError);
    fs::remove(path);
}

#if defined(EE_HAVE_ONNXRUNTIME)
// emotion2vec+ shared by the server's sessions: 8 users of one model (4 sessions' emotion engine
// and consistency check) estimate 1 s windows at the same moment, which batch together, and each
// gets what the model gives its window alone. EE_ACOUSTIC_MODEL_DIR=<a classifier export with a
// dynamic batch axis (ml/export/export_emotion2vec_onnx.py)>.
TEST(OnnxAcoustic, BatchesAcrossSessionsAndEachGetsItsOwn) {
    const char* dir = std::getenv("EE_ACOUSTIC_MODEL_DIR");
    if (dir == nullptr) GTEST_SKIP() << "set EE_ACOUSTIC_MODEL_DIR to run";
    Params alone;
    alone.set("acoustic", "onnx");
    alone.set("acoustic_model", dir);
    alone.set("acoustic_batch", "off");
    Params shared = alone;
    shared.set("acoustic_batch", "on");
    shared.set("sessions", "4");

    std::vector<std::vector<float>> windows(8, std::vector<float>(16000));
    for (std::size_t w = 0; w < windows.size(); ++w) {  // a different tone and loudness each
        for (std::size_t i = 0; i < windows[w].size(); ++i) {
            windows[w][i] = (0.05f + 0.05f * static_cast<float>(w)) *
                            std::sin(0.02f * static_cast<float>(w + 1) * static_cast<float>(i));
        }
    }
    const auto single = make_acoustic_model(alone, nullptr);
    std::vector<ModalityEstimate> expected;
    for (const auto& w : windows) expected.push_back(single->estimate(w, {}, 0.01f));

    std::vector<std::unique_ptr<IAcousticEmotionModel>> users;
    for (std::size_t i = 0; i < windows.size(); ++i) users.push_back(make_acoustic_model(shared, nullptr));
    std::vector<ModalityEstimate> got(windows.size());
    std::atomic<bool> go{false};
    std::vector<std::thread> threads;
    for (std::size_t i = 0; i < windows.size(); ++i) {
        threads.emplace_back([&, i] {
            while (!go.load()) std::this_thread::yield();
            got[i] = users[i]->estimate(windows[i], {}, 0.01f);
        });
    }
    go.store(true);
    for (auto& t : threads) t.join();
    for (std::size_t i = 0; i < windows.size(); ++i) {
        ASSERT_TRUE(got[i].valid) << i;
        EXPECT_NEAR(got[i].vad.v, expected[i].vad.v, 1e-4f) << i;
        EXPECT_NEAR(got[i].vad.a, expected[i].vad.a, 1e-4f) << i;
        EXPECT_NEAR(got[i].vad.d, expected[i].vad.d, 1e-4f) << i;
        EXPECT_NEAR(got[i].confidence.v, expected[i].confidence.v, 1e-4f) << i;
    }
}
#endif

}  // namespace
}  // namespace ee
