// What both streaming servers (5.3) ask of their clients: TLS files that go together, and a token
// compared without leaking where a guess went wrong.
#include <gtest/gtest.h>

#include <filesystem>

#include "core/runtime/params.hpp"
#include "core/server/security.hpp"

namespace ee {
namespace {

namespace fs = std::filesystem;

const fs::path kTls = fs::path(EE_SOURCE_DIR) / "tests" / "data" / "tls";

TEST(ServerSecurity, MatchesTokensExactly) {
    EXPECT_TRUE(token_matches("s3cret", "s3cret"));
    EXPECT_FALSE(token_matches("s3cret", "s3creT"));
    EXPECT_FALSE(token_matches("s3cret", "s3cre"));
    EXPECT_FALSE(token_matches("s3cret", "s3cret!"));
    EXPECT_FALSE(token_matches("s3cret", ""));
    EXPECT_FALSE(token_matches("", ""));  // no token set: nothing matches it
}

TEST(ServerSecurity, ReadsBearerHeaders) {
    EXPECT_EQ(bearer_token("Bearer abc.def"), "abc.def");
    EXPECT_EQ(bearer_token("bearer abc"), "abc");  // the scheme is case-insensitive
    EXPECT_EQ(bearer_token("Basic YWJj"), "");
    EXPECT_EQ(bearer_token("Bearer "), "");
    EXPECT_EQ(bearer_token(""), "");
}

TEST(ServerSecurity, KnowsWhichHostsOtherMachinesReach) {
    EXPECT_FALSE(reachable_beyond_localhost("127.0.0.1"));
    EXPECT_FALSE(reachable_beyond_localhost("localhost"));
    EXPECT_FALSE(reachable_beyond_localhost("::1"));
    EXPECT_TRUE(reachable_beyond_localhost("0.0.0.0"));
    EXPECT_TRUE(reachable_beyond_localhost("192.168.1.20"));
}

TEST(ServerSecurity, ValidatesTheTlsFiles) {
    ServerSecurity s;
    EXPECT_NO_THROW(s.validate());  // nothing asked
    EXPECT_FALSE(s.tls());
    s.cert_file = kTls / "server.pem";
    EXPECT_THROW(s.validate(), ConfigError);  // a certificate without its key
    s.key_file = kTls / "no-such.key";
    EXPECT_THROW(s.validate(), ConfigError);
    s.key_file = kTls / "server.key";
    EXPECT_NO_THROW(s.validate());
    EXPECT_TRUE(s.tls());
}

}  // namespace
}  // namespace ee
