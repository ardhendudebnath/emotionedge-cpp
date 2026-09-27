#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

namespace ee {

/// SHA-256 (FIPS 180-4), used to verify model files against models/manifest.json.
class Sha256 {
public:
    using Digest = std::array<std::uint8_t, 32>;

    Sha256() noexcept;
    void update(const void* data, std::size_t size) noexcept;
    void update(std::string_view data) noexcept { update(data.data(), data.size()); }
    /// Completes the hash; the object must be reset() before reuse.
    [[nodiscard]] Digest finish() noexcept;
    void reset() noexcept;

    [[nodiscard]] static std::string to_hex(const Digest& digest);

private:
    void compress(const std::uint8_t* block) noexcept;

    std::array<std::uint32_t, 8> state_{};
    std::array<std::uint8_t, 64> buffer_{};
    std::size_t buffered_ = 0;
    std::uint64_t total_bytes_ = 0;
};

[[nodiscard]] std::string sha256_hex(std::string_view data);
/// Hex digest of a file's contents, or nullopt if it cannot be read.
[[nodiscard]] std::optional<std::string> sha256_file(const std::filesystem::path& path);

}  // namespace ee
