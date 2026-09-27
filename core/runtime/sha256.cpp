#include "core/runtime/sha256.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <vector>

namespace ee {

namespace {

constexpr std::array<std::uint32_t, 64> kRound = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

constexpr std::uint32_t rotr(std::uint32_t x, int n) noexcept { return (x >> n) | (x << (32 - n)); }

}  // namespace

Sha256::Sha256() noexcept { reset(); }

void Sha256::reset() noexcept {
    state_ = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
              0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    buffered_ = 0;
    total_bytes_ = 0;
}

void Sha256::compress(const std::uint8_t* block) noexcept {
    std::array<std::uint32_t, 64> w{};
    for (int i = 0; i < 16; ++i) {
        w[i] = (std::uint32_t{block[4 * i]} << 24) | (std::uint32_t{block[4 * i + 1]} << 16) |
               (std::uint32_t{block[4 * i + 2]} << 8) | std::uint32_t{block[4 * i + 3]};
    }
    for (int i = 16; i < 64; ++i) {
        const std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        const std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    std::uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
    std::uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];
    for (int i = 0; i < 64; ++i) {
        const std::uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        const std::uint32_t ch = (e & f) ^ (~e & g);
        const std::uint32_t t1 = h + s1 + ch + kRound[i] + w[i];
        const std::uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        const std::uint32_t t2 = s0 + maj;
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
}

void Sha256::update(const void* data, std::size_t size) noexcept {
    const auto* p = static_cast<const std::uint8_t*>(data);
    total_bytes_ += size;
    if (buffered_ > 0) {
        const std::size_t take = std::min(size, buffer_.size() - buffered_);
        std::memcpy(buffer_.data() + buffered_, p, take);
        buffered_ += take;
        p += take;
        size -= take;
        if (buffered_ < buffer_.size()) return;
        compress(buffer_.data());
        buffered_ = 0;
    }
    while (size >= 64) {
        compress(p);
        p += 64;
        size -= 64;
    }
    if (size > 0) {
        std::memcpy(buffer_.data(), p, size);
        buffered_ = size;
    }
}

Sha256::Digest Sha256::finish() noexcept {
    const std::uint64_t bit_length = total_bytes_ * 8;
    buffer_[buffered_++] = 0x80;
    if (buffered_ > 56) {
        std::fill(buffer_.begin() + static_cast<std::ptrdiff_t>(buffered_), buffer_.end(), std::uint8_t{0});
        compress(buffer_.data());
        buffered_ = 0;
    }
    std::fill(buffer_.begin() + static_cast<std::ptrdiff_t>(buffered_), buffer_.begin() + 56, std::uint8_t{0});
    for (int i = 0; i < 8; ++i) buffer_[56 + i] = static_cast<std::uint8_t>(bit_length >> (56 - 8 * i));
    compress(buffer_.data());

    Digest out{};
    for (int i = 0; i < 8; ++i) {
        out[4 * i] = static_cast<std::uint8_t>(state_[i] >> 24);
        out[4 * i + 1] = static_cast<std::uint8_t>(state_[i] >> 16);
        out[4 * i + 2] = static_cast<std::uint8_t>(state_[i] >> 8);
        out[4 * i + 3] = static_cast<std::uint8_t>(state_[i]);
    }
    return out;
}

std::string Sha256::to_hex(const Digest& digest) {
    static constexpr char kHex[] = "0123456789abcdef";
    std::string out;
    out.reserve(64);
    for (std::uint8_t byte : digest) {
        out += kHex[byte >> 4];
        out += kHex[byte & 0x0f];
    }
    return out;
}

std::string sha256_hex(std::string_view data) {
    Sha256 h;
    h.update(data);
    return Sha256::to_hex(h.finish());
}

std::optional<std::string> sha256_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return std::nullopt;
    Sha256 h;
    std::vector<char> buf(1 << 20);
    while (in) {
        in.read(buf.data(), static_cast<std::streamsize>(buf.size()));
        const auto got = in.gcount();
        if (got > 0) h.update(buf.data(), static_cast<std::size_t>(got));
    }
    if (in.bad()) return std::nullopt;
    return Sha256::to_hex(h.finish());
}

}  // namespace ee
