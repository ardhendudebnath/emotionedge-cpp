#pragma once

#include <filesystem>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "core/translate/translator.hpp"

namespace ee {

/// Deterministic stand-in translator for tests and demos: exact sentence lookups from a YAML
/// phrasebook, with per-entry word links so emphasis spans move to the right target word.
/// Unknown sentences get a visible pseudo-translation ("[hi] ...") with markup preserved.
/// It is a fixture, not a translation model.
///
///     pairs:
///       en-hi:
///         - src: "I can't believe you did this!"
///           tgt: "मुझे यकीन नहीं हो रहा कि तुमने ऐसा किया!"
///           emphasis: { believe: यकीन }
class PhrasebookTranslator final : public ITranslator {
public:
    struct Entry {
        std::string target;
        std::map<std::string, std::string> emphasis;  ///< normalized source word -> target word
    };

    void add(std::string_view source_language, std::string_view target_language, std::string_view source,
             Entry entry);
    [[nodiscard]] static PhrasebookTranslator load(const std::filesystem::path& path);

    [[nodiscard]] TranslationResult translate(const TranslationRequest& request) override;
    [[nodiscard]] bool understands_control_tokens() const noexcept override { return true; }
    [[nodiscard]] bool preserves_markup() const noexcept override { return true; }

    /// Lowercase, apostrophes unified, whitespace collapsed.
    [[nodiscard]] static std::string normalize(std::string_view text);

private:
    std::map<std::string, Entry, std::less<>> entries_;  // key: "<src>-<tgt>|<normalized source>"
};

}  // namespace ee
