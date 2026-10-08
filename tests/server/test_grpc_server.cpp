// The gRPC API (5.3) over a real channel: a generated client streams the blueprint walkthrough
// and gets typed results back. Stand-in engines (config/pipeline.yaml).
#include <gtest/gtest.h>

#include <grpcpp/grpcpp.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "core/pipeline/demo.hpp"
#include "core/server/grpc_server.hpp"
#include "core/server/session_pool.hpp"
#include "emotionedge/v1/translator.grpc.pb.h"

namespace ee {
namespace {

namespace fs = std::filesystem;
namespace pb = emotionedge::v1;
using json = nlohmann::json;

SessionPool::Options pool_options(const DemoInput& demo, const std::string& name) {
    const fs::path dir = fs::temp_directory_path() / ("ee_grpc_" + name);
    fs::create_directories(dir);
    std::ofstream(dir / "script.json", std::ios::binary) << demo.script_json;
    SessionPool::Options o;
    o.config = fs::path(EE_SOURCE_DIR) / "config" / "pipeline.yaml";
    o.overrides = {{"asr.engine", "scripted"}, {"asr.script", (dir / "script.json").string()},
                   {"state.join_timeout_ms", "20000"}, {"emotion.transcript_timeout_ms", "20000"}};
    return o;
}

std::string pcm16(const std::vector<float>& audio, std::size_t from, std::size_t to) {
    std::string bytes;
    for (std::size_t i = from; i < std::min(to, audio.size()); ++i) {
        const auto s = static_cast<std::uint16_t>(static_cast<std::int16_t>(std::clamp(audio[i], -1.0f, 1.0f) * 32767.0f));
        bytes.push_back(static_cast<char>(s & 0xFFu));
        bytes.push_back(static_cast<char>(s >> 8));
    }
    return bytes;
}

TEST(GrpcServer, TranslatesAStreamIntoTypedResults) {
    const DemoInput demo = make_walkthrough_input();
    SessionPool pool(pool_options(demo, "stream"));
    pool.start();
    GrpcServer server(pool, "127.0.0.1", 0);
    server.start();
    ASSERT_GT(server.port(), 0);

    auto channel = grpc::CreateChannel("127.0.0.1:" + std::to_string(server.port()), grpc::InsecureChannelCredentials());
    auto stub = pb::Translator::NewStub(channel);
    grpc::ClientContext context;
    auto call = stub->Translate(&context);

    pb::TranslateRequest request;
    request.mutable_config()->set_sample_rate(demo.sample_rate);
    ASSERT_TRUE(call->Write(request));
    const std::size_t block = static_cast<std::size_t>(demo.sample_rate / 50);
    for (std::size_t pos = 0; pos < demo.audio.size(); pos += block) {
        request.set_audio(pcm16(demo.audio, pos, pos + block));
        ASSERT_TRUE(call->Write(request));
    }
    request.mutable_end();
    ASSERT_TRUE(call->Write(request));
    call->WritesDone();

    std::vector<pb::TranslateResponse> responses;
    pb::TranslateResponse response;
    std::size_t audio_bytes = 0;
    while (call->Read(&response)) {
        if (response.response_case() == pb::TranslateResponse::kAudio) {
            audio_bytes += response.audio().size();
            continue;
        }
        responses.push_back(response);
    }
    const grpc::Status status = call->Finish();
    ASSERT_TRUE(status.ok()) << status.error_message();

    ASSERT_FALSE(responses.empty());
    ASSERT_EQ(responses.front().response_case(), pb::TranslateResponse::kReady);
    EXPECT_EQ(responses.front().ready().input_rate(), demo.sample_rate);
    EXPECT_EQ(responses.front().ready().target_language(), "hi");
    ASSERT_EQ(responses.back().response_case(), pb::TranslateResponse::kDone);
    const json session = json::parse(responses.back().done().session_json());
    EXPECT_EQ(session["utterances"].size(), 1u);

    const auto find = [&](pb::TranslateResponse::ResponseCase kind, bool final_only) -> const pb::TranslateResponse* {
        for (const auto& r : responses) {
            if (r.response_case() != kind) continue;
            if (final_only && kind == pb::TranslateResponse::kTranscript && !r.transcript().is_final()) continue;
            if (final_only && kind == pb::TranslateResponse::kTranslation && !r.translation().is_final()) continue;
            return &r;
        }
        return nullptr;
    };
    const auto* heard = find(pb::TranslateResponse::kTranscript, true);
    ASSERT_NE(heard, nullptr);
    EXPECT_EQ(heard->transcript().text(), "I can't believe you did this!");
    const auto* felt = find(pb::TranslateResponse::kEmotion, false);
    ASSERT_NE(felt, nullptr);
    EXPECT_EQ(felt->emotion().point().label(), "anger");
    EXPECT_LT(felt->emotion().point().valence(), 0.0f);
    ASSERT_GE(felt->emotion().emphasis_size(), 1);
    EXPECT_EQ(felt->emotion().emphasis(0), "believe");
    const auto* translated = find(pb::TranslateResponse::kTranslation, true);
    ASSERT_NE(translated, nullptr);
    EXPECT_EQ(translated->translation().text(), "मुझे यकीन नहीं हो रहा कि तुमने ऐसा किया!");
    EXPECT_EQ(translated->translation().utterance(), heard->transcript().utterance());
    const auto* planned = find(pb::TranslateResponse::kProsody, false);
    ASSERT_NE(planned, nullptr);
    EXPECT_GT(planned->prosody().pitch_pct(), 0.0f);
    const auto* scored = find(pb::TranslateResponse::kConsistency, false);
    ASSERT_NE(scored, nullptr);
    EXPECT_FALSE(scored->consistency().heard().label().empty());
    EXPECT_NE(find(pb::TranslateResponse::kPlayout, false), nullptr);
    EXPECT_GT(audio_bytes, static_cast<std::size_t>(24000 * 2 / 2));  // over 0.5 s of speech

    pool.stop();
    server.stop();
}

TEST(GrpcServer, RefusesBeyondTheLimitWithResourceExhausted) {
    const DemoInput demo = make_walkthrough_input();
    SessionPool pool(pool_options(demo, "limit"));  // max_sessions 1
    pool.start();
    GrpcServer server(pool, "127.0.0.1", 0);
    server.start();
    auto stub = pb::Translator::NewStub(
        grpc::CreateChannel("127.0.0.1:" + std::to_string(server.port()), grpc::InsecureChannelCredentials()));

    grpc::ClientContext first_context;
    auto first = stub->Translate(&first_context);
    pb::TranslateRequest config;
    config.mutable_config()->set_sample_rate(16000);
    ASSERT_TRUE(first->Write(config));
    pb::TranslateResponse response;
    ASSERT_TRUE(first->Read(&response));  // Ready: the one session is taken
    ASSERT_EQ(response.response_case(), pb::TranslateResponse::kReady);

    grpc::ClientContext second_context;
    auto second = stub->Translate(&second_context);
    ASSERT_TRUE(second->Write(config));
    second->WritesDone();
    while (second->Read(&response)) {
    }
    const grpc::Status refused = second->Finish();
    EXPECT_EQ(refused.error_code(), grpc::StatusCode::RESOURCE_EXHAUSTED);

    first_context.TryCancel();  // the first client hangs up
    EXPECT_EQ(first->Finish().error_code(), grpc::StatusCode::CANCELLED);
    pool.stop();
    server.stop();
}

}  // namespace
}  // namespace ee
