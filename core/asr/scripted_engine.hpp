#pragma once

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "core/asr/asr_engine.hpp"

namespace ee {

/// Deterministic stand-in recognizer for tests, demos and benchmarks without model files. It
/// "hears" words from a transcript script with absolute timings (input-timeline seconds):
///
///     {"language": "en", "words": [{"w": "I", "t0": 0.40, "t1": 0.52}, ...]}
///
/// A partial request returns the words that ended within the audio decoded so far; a final
/// request returns every word whose midpoint falls inside the utterance.
class ScriptedAsrEngine final : public IAsrEngine {
public:
    struct ScriptWord {
        std::string text;
        double t0 = 0.0;
        double t1 = 0.0;
    };

    ScriptedAsrEngine(std::vector<ScriptWord> words, std::string language);
    [[nodiscard]] static ScriptedAsrEngine from_json(std::string_view json);
    [[nodiscard]] static ScriptedAsrEngine from_file(const std::filesystem::path& path);

    [[nodiscard]] AsrResult transcribe(const AsrRequest& request) override;

private:
    std::vector<ScriptWord> words_;
    std::string language_;
};

/// Serializes words with absolute times into the script format above.
[[nodiscard]] std::string make_asr_script(const std::vector<Word>& words, double offset_s, std::string_view language);

}  // namespace ee
