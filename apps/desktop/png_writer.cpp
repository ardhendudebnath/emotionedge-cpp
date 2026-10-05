#include "apps/desktop/png_writer.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <string>

namespace ee::desktop {

namespace {

std::uint32_t crc32(const unsigned char* data, std::size_t n, std::uint32_t crc = 0) {
    static const std::array<std::uint32_t, 256> table = [] {
        std::array<std::uint32_t, 256> t{};
        for (std::uint32_t i = 0; i < 256; ++i) {
            std::uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1u) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            t[i] = c;
        }
        return t;
    }();
    crc = ~crc;
    for (std::size_t i = 0; i < n; ++i) crc = table[(crc ^ data[i]) & 0xFFu] ^ (crc >> 8);
    return ~crc;
}

void put32(std::string& out, std::uint32_t v) {
    for (int shift = 24; shift >= 0; shift -= 8) out.push_back(static_cast<char>((v >> shift) & 0xFFu));
}

void chunk(std::ofstream& f, const char* type, const std::string& data) {
    std::string out;
    put32(out, static_cast<std::uint32_t>(data.size()));
    const std::string body = std::string(type, 4) + data;
    out += body;
    put32(out, crc32(reinterpret_cast<const unsigned char*>(body.data()), body.size()));
    f.write(out.data(), static_cast<std::streamsize>(out.size()));
}

}  // namespace

bool write_png(const std::filesystem::path& path, const std::vector<unsigned char>& rgba, int width, int height) {
    if (width <= 0 || height <= 0 || rgba.size() < static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4) {
        return false;
    }
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write("\x89PNG\r\n\x1a\n", 8);

    std::string ihdr;
    put32(ihdr, static_cast<std::uint32_t>(width));
    put32(ihdr, static_cast<std::uint32_t>(height));
    ihdr += std::string("\x08\x06\x00\x00\x00", 5);  // 8-bit RGBA, deflate, no filter, no interlace
    chunk(f, "IHDR", ihdr);

    // Each row: filter type 0, then the pixels.
    std::string raw;
    const std::size_t stride = static_cast<std::size_t>(width) * 4;
    raw.reserve((stride + 1) * static_cast<std::size_t>(height));
    for (int y = 0; y < height; ++y) {
        raw.push_back('\0');
        raw.append(reinterpret_cast<const char*>(rgba.data()) + static_cast<std::size_t>(y) * stride, stride);
    }
    std::string zlib = "\x78\x01";
    std::uint32_t a = 1, b = 0;
    for (const char c : raw) {
        a = (a + static_cast<unsigned char>(c)) % 65521u;
        b = (b + a) % 65521u;
    }
    for (std::size_t pos = 0; pos < raw.size() || pos == 0; pos += 65535) {
        const std::size_t len = std::min<std::size_t>(65535, raw.size() - pos);
        zlib.push_back(pos + len >= raw.size() ? '\x01' : '\x00');  // last block?
        zlib.push_back(static_cast<char>(len & 0xFFu));
        zlib.push_back(static_cast<char>((len >> 8) & 0xFFu));
        zlib.push_back(static_cast<char>(~len & 0xFFu));
        zlib.push_back(static_cast<char>((~len >> 8) & 0xFFu));
        zlib.append(raw, pos, len);
        if (len == 0) break;
    }
    put32(zlib, (b << 16) | a);
    chunk(f, "IDAT", zlib);
    chunk(f, "IEND", "");
    return static_cast<bool>(f);
}

}  // namespace ee::desktop
