#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>

#include "core/runtime/sha256.hpp"

namespace ee {
namespace {

// FIPS 180-4 / NIST CAVP example vectors.
TEST(Sha256, KnownVectors) {
    EXPECT_EQ(sha256_hex(""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    EXPECT_EQ(sha256_hex("abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    EXPECT_EQ(sha256_hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
              "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    EXPECT_EQ(sha256_hex(std::string(1'000'000, 'a')),
              "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}

TEST(Sha256, ChunkedUpdatesMatchOneShot) {
    std::string data;
    for (int i = 0; i < 1000; ++i) data += static_cast<char>('a' + i % 26);
    for (std::size_t chunk : {1u, 3u, 55u, 56u, 63u, 64u, 65u, 127u}) {
        Sha256 h;
        for (std::size_t pos = 0; pos < data.size(); pos += chunk) {
            h.update(std::string_view(data).substr(pos, chunk));
        }
        EXPECT_EQ(Sha256::to_hex(h.finish()), sha256_hex(data)) << "chunk " << chunk;
    }
}

TEST(Sha256, HashesFiles) {
    const auto path = std::filesystem::temp_directory_path() / "ee_sha256_test.bin";
    {
        std::ofstream out(path, std::ios::binary);
        out << "abc";
    }
    const auto digest = sha256_file(path);
    ASSERT_TRUE(digest.has_value());
    EXPECT_EQ(*digest, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    std::filesystem::remove(path);
    EXPECT_FALSE(sha256_file(path).has_value());
}

}  // namespace
}  // namespace ee
