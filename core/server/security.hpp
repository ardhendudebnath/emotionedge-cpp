#pragma once

#include <filesystem>
#include <string>
#include <string_view>

namespace ee {

/// What a streaming server (5.3) asks of its clients: TLS, and a shared bearer token.
struct ServerSecurity {
    std::filesystem::path cert_file;  ///< PEM certificate (chain); with key_file: TLS (wss://, gRPC SSL)
    std::filesystem::path key_file;   ///< PEM private key
    std::string token;                ///< clients must present it; empty = no token asked

    [[nodiscard]] bool tls() const noexcept { return !cert_file.empty(); }
    /// Throws ConfigError when only one of the TLS files is given, or one cannot be read.
    void validate() const;
};

/// Whether `presented` is the expected token, in time that does not depend on where they differ.
[[nodiscard]] bool token_matches(std::string_view expected, std::string_view presented) noexcept;

/// The token in an "Authorization: Bearer <token>" value; empty if it is not one.
[[nodiscard]] std::string_view bearer_token(std::string_view authorization) noexcept;

/// Whether a server listening on `host` can be reached from other machines.
[[nodiscard]] bool reachable_beyond_localhost(std::string_view host) noexcept;

/// A whole file (PEM material).
[[nodiscard]] std::string read_text_file(const std::filesystem::path& path);

}  // namespace ee
