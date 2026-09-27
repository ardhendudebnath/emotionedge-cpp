// Kokoro's G2P front end: the port of misaki's EspeakG2P (on phonemizer) must produce the exact
// phoneme strings Kokoro-82M was trained on.
#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>

#include "core/tts/kokoro_g2p.hpp"
#if defined(EE_HAVE_PIPER)
#include "core/tts/espeak.hpp"
#endif

namespace ee::kokoro {
namespace {

TEST(KokoroG2p, PreservesAndRestoresPunctuationLikePhonemizer) {
    const PunctuationSplit split = preserve_punctuation("hello, my world!");
    EXPECT_EQ(split.chunks, (std::vector<std::string>{"hello", "my world"}));
    ASSERT_EQ(split.marks.size(), 2u);
    EXPECT_EQ(split.marks[0].text, ", ");
    EXPECT_EQ(split.marks[0].position, 'I');
    EXPECT_EQ(split.marks[1].text, "!");
    EXPECT_EQ(split.marks[1].position, 'E');
    EXPECT_EQ(restore_punctuation({"h1 ", "m1 w1 "}, split.marks), "h1, m1 w1! ");

    // A leading mark is B; empty chunks are dropped; a line of marks only is A.
    const PunctuationSplit quoted = preserve_punctuation("\xC2\xAB" "ab\xC2\xBB c");  // «ab» c
    EXPECT_EQ(quoted.chunks, (std::vector<std::string>{"ab", "c"}));
    ASSERT_EQ(quoted.marks.size(), 2u);
    EXPECT_EQ(quoted.marks[0].position, 'B');
    EXPECT_EQ(quoted.marks[1].text, "\xC2\xBB ");
    const PunctuationSplit alone = preserve_punctuation("...");
    EXPECT_TRUE(alone.chunks.empty());
    ASSERT_EQ(alone.marks.size(), 1u);
    EXPECT_EQ(alone.marks[0].position, 'A');
    EXPECT_EQ(restore_punctuation({}, alone.marks), "...");
    // No marks: the chunk is the line (the danda is not a phonemizer mark).
    EXPECT_EQ(preserve_punctuation("\xE0\xA4\xA0\xE0\xA5\x80\xE0\xA4\x95 \xE0\xA5\xA4").chunks.size(), 1u);
}

TEST(KokoroG2p, PostprocessesEspeakLines) {
    // Separators merged, language-switch flags removed, ties become '^', words end with ' '.
    EXPECT_EQ(postprocess_espeak_line("  h\xC9\x99l\xCB\x88o_ (en)w\xCB\x88\xC9\x9C\xCB\x90ld(hi)\n"),
              "h\xC9\x99l\xCB\x88o w\xCB\x88\xC9\x9C\xCB\x90ld ");
    EXPECT_EQ(postprocess_espeak_line("t\xCD\xA1\xCA\x83" "a\xCB\x90"), "t^\xCA\x83" "a\xCB\x90 ");  // t͡ʃaː
    EXPECT_EQ(postprocess_espeak_line("(en)"), "");
}

TEST(KokoroG2p, MergesTiesAndRestoresParentheses) {
    // "(a)" -> «a» -> espeak "t͡ʃaː" -> «t^ʃaː» -> merge t^ʃ -> ʧ -> (ʧaː)
    const auto fake = [](const std::string& chunk) {
        return chunk == "a" ? std::string("t\xCD\xA1\xCA\x83" "a\xCB\x90") : chunk;
    };
    EXPECT_EQ(misaki_g2p("(a)", fake), "(\xCA\xA7" "a\xCB\x90)");
    // Hyphens and remaining ties are dropped.
    EXPECT_EQ(misaki_g2p("x", [](const std::string&) { return std::string("k-a\xCD\xA1" "b"); }), "kab");
}

#if defined(EE_HAVE_PIPER)
// The whole G2P through espeak-ng against misaki's output (ml/export/export_kokoro_onnx.py
// --golden, made with the same espeak-ng library). EE_ESPEAK_DATA points at espeak-ng-data when
// it is not in the library's default place.
TEST(KokoroG2p, MatchesMisakiOnHindi) {
    const char* data = std::getenv("EE_ESPEAK_DATA");
    espeak::initialize(data != nullptr ? data : "");
    std::ifstream in(std::filesystem::path(EE_SOURCE_DIR) / "tests" / "golden" / "kokoro_g2p_hi.json");
    const nlohmann::json golden = nlohmann::json::parse(in);
    ASSERT_GE(golden.size(), 10u);
    for (const auto& item : golden) {
        const std::string text = item.at("text").get<std::string>();
        const std::string got =
            misaki_g2p(text, [](const std::string& chunk) { return espeak::text_to_phonemes(chunk, "hi", true); });
        EXPECT_EQ(got, item.at("phonemes").get<std::string>()) << text;
    }
}
#endif

}  // namespace
}  // namespace ee::kokoro
