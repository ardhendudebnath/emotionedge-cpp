#include "core/server/security.hpp"

#include <algorithm>
#include <fstream>
#include <iterator>

#include "core/runtime/params.hpp"  // ConfigError

namespace ee {

void ServerSecurity::validate() const {
    if (cert_file.empty() != key_file.empty()) {
        throw ConfigError("TLS needs both a certificate and its private key (--tls-cert and --tls-key)");
    }
    if (tls()) {
        (void)read_text_file(cert_file);
        (void)read_text_file(key_file);
    }
}

bool token_matches(std::string_view expected, std::string_view presented) noexcept {
    // Every byte of the longer string is visited whatever the content, so the time taken says
    // nothing about how much of a guess was right.
    const std::size_t n = std::max(expected.size(), presented.size());
    unsigned diff = expected.size() == presented.size() ? 0u : 1u;
    for (std::size_t i = 0; i < n; ++i) {
        const auto a = static_cast<unsigned char>(i < expected.size() ? expected[i] : 0);
        const auto b = static_cast<unsigned char>(i < presented.size() ? presented[i] : 0);
        diff |= static_cast<unsigned>(a ^ b);
    }
    return diff == 0 && !expected.empty();
}

std::string_view bearer_token(std::string_view authorization) noexcept {
    constexpr std::string_view kScheme = "Bearer ";
    if (authorization.size() <= kScheme.size()) return {};
    for (std::size_t i = 0; i < kScheme.size(); ++i) {  // the scheme is case-insensitive
        const char c = authorization[i];
        const char lower = (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
        const char want = (kScheme[i] >= 'A' && kScheme[i] <= 'Z') ? static_cast<char>(kScheme[i] - 'A' + 'a') : kScheme[i];
        if (lower != want) return {};
    }
    return authorization.substr(kScheme.size());
}

bool reachable_beyond_localhost(std::string_view host) noexcept {
    return !(host == "127.0.0.1" || host == "localhost" || host == "::1" || host == "[::1]");
}

std::string read_text_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw ConfigError("cannot read '" + path.string() + "'");
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

}  // namespace ee
