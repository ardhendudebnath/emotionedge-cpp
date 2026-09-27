#pragma once

#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <string_view>

namespace ee {

/// Speech register used in the MT control tokens (`reg=casual`).
enum class Register { Casual, Formal, Neutral };

[[nodiscard]] std::string_view to_string(Register r) noexcept;
[[nodiscard]] std::optional<Register> parse_register(std::string_view name) noexcept;

/// Blueprint 3.3 "Expressivity Profiles": how strongly the target language's speaking norms
/// carry emotion. The controller scales its prosody plan by these, and translation takes the
/// default register from them. Values are v1 placeholders to calibrate with ECS data and
/// listening tests (P5), not measurements.
struct ExpressivityProfile {
    std::string language;
    float intensity = 1.0f;    ///< scales the whole prosody plan
    float pitch_range = 1.0f;  ///< extra scale on pitch-range changes (tonal languages < 1)
    float rate = 1.0f;
    float energy = 1.0f;
    Register reg = Register::Casual;
    float length_ratio = 1.0f;  ///< target/source length in words, for wait-k budgets
};

class ExpressivityProfiles {
public:
    /// Built-in defaults (mirrored in config/expressivity.yaml).
    [[nodiscard]] static ExpressivityProfiles defaults();
    /// YAML: `default: {...}` and `languages: {hi: {...}, ...}`. Throws ConfigError.
    [[nodiscard]] static ExpressivityProfiles load(const std::filesystem::path& path);

    /// The profile for `language`, or the default profile.
    [[nodiscard]] ExpressivityProfile get(std::string_view language) const;

private:
    ExpressivityProfile fallback_;
    std::map<std::string, ExpressivityProfile, std::less<>> languages_;
};

}  // namespace ee
