#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include <string>

#include "core/pipeline/recorder.hpp"
#include "core/runtime/config.hpp"
#include "core/runtime/utf8.hpp"

namespace ee {
namespace {

TEST(Utf8, ValidTextIsUnchanged) {
    const std::string text = "Kids are talking. मुझे यकीन नहीं! ケーズ 😀";
    EXPECT_EQ(valid_utf8(text), text);
    EXPECT_EQ(valid_utf8(""), "");
}

TEST(Utf8, DropsWhatIsNotUtf8) {
    EXPECT_EQ(valid_utf8("\xE3\x81."), ".");          // babble's half character, then a full stop
    EXPECT_EQ(valid_utf8("ab\xE3\x81"), "ab");        // cut short at the end
    EXPECT_EQ(valid_utf8("a\x80" "b"), "ab");         // a stray continuation byte
    EXPECT_EQ(valid_utf8("\xC0\xAF"), "");            // an overlong '/'
    EXPECT_EQ(valid_utf8("\xED\xA0\x80x"), "x");      // a UTF-16 surrogate
    EXPECT_EQ(valid_utf8("\xF4\x90\x80\x80y"), "y");  // past U+10FFFF
    EXPECT_EQ(valid_utf8("\xFFz"), "z");              // no lead byte at all
}

// The session's JSON export took the session down on half a character (babble, 20 dB): it now
// writes U+FFFD in its place, as the live events already did.
TEST(Utf8, SessionExportSurvivesInvalidText) {
    UtteranceRecord record;
    record.source_text = "\xE3\x81.";
    record.translation = "ok";
    PipelineSpec spec;
    std::string text;
    ASSERT_NO_THROW(text = to_session_json({record}, spec, nullptr));
    const nlohmann::json session = nlohmann::json::parse(text);
    const std::string source = session["utterances"][0]["source"]["text"];
    EXPECT_NE(source.find("\xEF\xBF\xBD"), std::string::npos);  // U+FFFD where the half character was
    EXPECT_EQ(source.back(), '.');
    EXPECT_EQ(valid_utf8(source), source);
}

}  // namespace
}  // namespace ee
