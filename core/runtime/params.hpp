#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ee {

/// Invalid or missing configuration. Thrown during setup, never from a stage's hot path.
struct ConfigError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

/// Flat key/value parameters for one stage. Nested YAML maps become dotted keys
/// (`agc.target_dbfs`) and sequences become indexed keys (`kinds.0`, `kinds.1`), which keeps
/// stage code independent of the YAML library.
class Params {
public:
    using Map = std::map<std::string, std::string, std::less<>>;

    void set(std::string key, std::string value);
    [[nodiscard]] bool has(std::string_view key) const;
    [[nodiscard]] bool empty() const noexcept { return values_.empty(); }
    [[nodiscard]] const Map& items() const noexcept { return values_; }

    [[nodiscard]] std::string str(std::string_view key, std::string_view fallback = {}) const;
    /// Numeric getters throw ConfigError when the value is present but not a number.
    [[nodiscard]] double number(std::string_view key, double fallback) const;
    [[nodiscard]] float real(std::string_view key, float fallback) const;
    [[nodiscard]] std::int64_t integer(std::string_view key, std::int64_t fallback) const;
    [[nodiscard]] bool flag(std::string_view key, bool fallback) const;
    /// `key.0`, `key.1`, ... or, if only `key` is set, a single-element list.
    [[nodiscard]] std::vector<std::string> list(std::string_view key) const;
    /// Keys under `prefix.`, with the prefix stripped.
    [[nodiscard]] Params sub(std::string_view prefix) const;

private:
    Map values_;
};

}  // namespace ee
