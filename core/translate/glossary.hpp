#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace ee {

/// Blueprint 3.2 "Glossary / term lock". Terms are swapped for placeholders (`__T0__`) before
/// translation and restored afterwards, so names and product terms survive MT unchanged (or as
/// their fixed translation).
class Glossary {
public:
    struct Term {
        std::string source;
        std::string target;  ///< empty = keep the source term verbatim
    };
    struct Protected {
        std::string text;
        std::vector<std::string> replacements;  ///< placeholder i -> text to restore
    };

    void add(std::string source, std::string target = {});
    [[nodiscard]] bool empty() const noexcept { return terms_.empty(); }

    /// YAML mapping `{term: translation}`; a null translation keeps the term. Throws ConfigError.
    [[nodiscard]] static Glossary load(const std::filesystem::path& path);

    /// Replaces whole-word, case-insensitive occurrences (longest terms first).
    [[nodiscard]] Protected protect(std::string_view text) const;
    /// Puts the terms back. Placeholders the MT dropped are counted in `missing` if given.
    [[nodiscard]] std::string restore(std::string_view translated, const Protected& p,
                                      std::uint32_t* missing = nullptr) const;

private:
    std::vector<Term> terms_;
};

}  // namespace ee
