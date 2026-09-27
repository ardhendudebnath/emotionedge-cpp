#include "core/runtime/params.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstdlib>

namespace ee {

namespace {
std::string lower(std::string_view s) {
    std::string out(s);
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return out;
}
}  // namespace

void Params::set(std::string key, std::string value) { values_[std::move(key)] = std::move(value); }

bool Params::has(std::string_view key) const { return values_.find(key) != values_.end(); }

std::string Params::str(std::string_view key, std::string_view fallback) const {
    const auto it = values_.find(key);
    return it != values_.end() ? it->second : std::string(fallback);
}

double Params::number(std::string_view key, double fallback) const {
    const auto it = values_.find(key);
    if (it == values_.end()) return fallback;
    const std::string& s = it->second;
    char* end = nullptr;
    const double v = std::strtod(s.c_str(), &end);
    if (s.empty() || end != s.c_str() + s.size()) {
        throw ConfigError("parameter '" + std::string(key) + "' is not a number: '" + s + "'");
    }
    return v;
}

float Params::real(std::string_view key, float fallback) const {
    return static_cast<float>(number(key, fallback));
}

std::int64_t Params::integer(std::string_view key, std::int64_t fallback) const {
    const auto it = values_.find(key);
    if (it == values_.end()) return fallback;
    const std::string& s = it->second;
    std::int64_t v = 0;
    const auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
    if (ec != std::errc{} || ptr != s.data() + s.size()) {
        throw ConfigError("parameter '" + std::string(key) + "' is not an integer: '" + s + "'");
    }
    return v;
}

bool Params::flag(std::string_view key, bool fallback) const {
    const auto it = values_.find(key);
    if (it == values_.end()) return fallback;
    const std::string v = lower(it->second);
    if (v == "true" || v == "yes" || v == "on" || v == "1") return true;
    if (v == "false" || v == "no" || v == "off" || v == "0") return false;
    throw ConfigError("parameter '" + std::string(key) + "' is not a boolean: '" + it->second + "'");
}

std::vector<std::string> Params::list(std::string_view key) const {
    std::vector<std::string> out;
    const std::string base(key);
    for (std::size_t i = 0;; ++i) {
        const auto it = values_.find(base + "." + std::to_string(i));
        if (it == values_.end()) break;
        out.push_back(it->second);
    }
    if (out.empty()) {
        if (const auto it = values_.find(key); it != values_.end() && !it->second.empty()) {
            out.push_back(it->second);
        }
    }
    return out;
}

Params Params::sub(std::string_view prefix) const {
    Params out;
    const std::string p = std::string(prefix) + ".";
    for (auto it = values_.lower_bound(p); it != values_.end(); ++it) {
        if (it->first.compare(0, p.size(), p) != 0) break;
        out.set(it->first.substr(p.size()), it->second);
    }
    return out;
}

}  // namespace ee
