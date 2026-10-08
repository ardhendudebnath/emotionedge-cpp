// The WebSocket server (5.3) end to end over a real socket: a client streams the blueprint
// walkthrough's speech in and gets the live events and the translated speech back. Stand-in
// engines (config/pipeline.yaml), so no model files are needed.
#include <gtest/gtest.h>

#include <ixwebsocket/IXNetSystem.h>
#include <ixwebsocket/IXWebSocket.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "core/audio/resampler.hpp"
#include "core/pipeline/demo.hpp"
#include "core/server/session_pool.hpp"
#include "core/server/websocket_server.hpp"

namespace ee {
namespace {

namespace fs = std::filesystem;
using json = nlohmann::json;
using namespace std::chrono_literals;

/// A session pool and a WebSocket server on it (the server goes first: it closes its sessions).
struct Served {
    std::unique_ptr<SessionPool> pool;
    std::unique_ptr<WebSocketServer> server;
    int port = 0;
};

/// A server on the first free port from 18765, with the stand-in engines and a word script for
/// the scripted ASR.
Served start_server(const fs::path& script, std::size_t max_sessions, std::size_t warm_sessions = 1) {
    SessionPool::Options o;
    o.config = fs::path(EE_SOURCE_DIR) / "config" / "pipeline.yaml";
    o.max_sessions = max_sessions;
    o.warm_sessions = warm_sessions;
    // Long join timeouts: sanitizer builds must not take the "latest estimate" fallback.
    o.overrides = {{"asr.engine", "scripted"}, {"asr.script", script.string()},
                   {"state.join_timeout_ms", "20000"}, {"emotion.transcript_timeout_ms", "20000"}};
    Served s;
    s.pool = std::make_unique<SessionPool>(o);
    s.pool->start();
    for (s.port = 18765; s.port < 18805; ++s.port) {
        auto server = std::make_unique<WebSocketServer>(*s.pool, "127.0.0.1", s.port);
        try {
            server->start();
            s.server = std::move(server);
            return s;
        } catch (const std::runtime_error&) {
        }
    }
    return s;
}

/// A WebSocket client that records what the server sends.
class Client {
public:
    explicit Client(const std::string& url) {
        ix::initNetSystem();
        ws_.setUrl(url);
        ws_.disableAutomaticReconnection();
        ws_.setOnMessageCallback([this](const ix::WebSocketMessagePtr& msg) {
            const std::lock_guard<std::mutex> lock(mu_);
            if (msg->type == ix::WebSocketMessageType::Message) {
                if (msg->binary) {
                    audio_bytes_ += msg->str.size();
                } else {
                    events_.push_back(json::parse(msg->str));
                }
            } else if (msg->type == ix::WebSocketMessageType::Close) {
                closed_ = true;
            }
            cv_.notify_all();
        });
        ws_.start();
    }
    ~Client() { ws_.stop(); }

    /// Waits until an event of `type` arrives; false on timeout.
    bool wait_for(const std::string& type, std::chrono::seconds timeout) {
        std::unique_lock<std::mutex> lock(mu_);
        return cv_.wait_for(lock, timeout, [&] {
            for (const json& e : events_) {
                if (e["type"] == type) return true;
            }
            return closed_ && type == "close";
        });
    }
    void send_pcm(const std::vector<float>& audio, int rate) {
        const std::size_t block = static_cast<std::size_t>(rate / 50);  // 20 ms
        for (std::size_t pos = 0; pos < audio.size(); pos += block) {
            std::string pcm;
            for (std::size_t i = pos; i < std::min(audio.size(), pos + block); ++i) {
                const auto s = static_cast<std::int16_t>(std::clamp(audio[i], -1.0f, 1.0f) * 32767.0f);
                pcm.push_back(static_cast<char>(static_cast<std::uint16_t>(s) & 0xFFu));
                pcm.push_back(static_cast<char>(static_cast<std::uint16_t>(s) >> 8));
            }
            ws_.sendBinary(pcm);
        }
    }
    void send_text(const std::string& text) { ws_.sendText(text); }
    [[nodiscard]] std::vector<json> events() {
        const std::lock_guard<std::mutex> lock(mu_);
        return events_;
    }
    [[nodiscard]] std::size_t audio_bytes() {
        const std::lock_guard<std::mutex> lock(mu_);
        return audio_bytes_;
    }

private:
    ix::WebSocket ws_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::vector<json> events_;
    std::size_t audio_bytes_ = 0;
    bool closed_ = false;
};

template <typename Pred>
bool eventually(Pred pred, std::chrono::seconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (!pred()) {
        if (std::chrono::steady_clock::now() > deadline) return false;
        std::this_thread::sleep_for(20ms);
    }
    return true;
}

fs::path write_script(const DemoInput& demo, const std::string& name) {
    const fs::path dir = fs::temp_directory_path() / ("ee_server_" + name);
    fs::create_directories(dir);
    std::ofstream(dir / "script.json", std::ios::binary) << demo.script_json;
    return dir / "script.json";
}

TEST(WebSocketServer, StreamsEventsAndTranslatedSpeech) {
    const DemoInput demo = make_walkthrough_input();
    const Served s = start_server(write_script(demo, "stream"), 1);
    ASSERT_NE(s.server, nullptr) << "no free port in 18765-18804";

    Client client("ws://127.0.0.1:" + std::to_string(s.port) + "/?rate=" + std::to_string(demo.sample_rate));
    ASSERT_TRUE(client.wait_for("ready", 60s));
    client.send_pcm(demo.audio, demo.sample_rate);
    client.send_text(R"({"type": "end"})");
    ASSERT_TRUE(client.wait_for("done", 120s));

    const std::vector<json> events = client.events();
    const auto find = [&](const std::string& type, bool final_only = false) -> const json* {
        for (const json& e : events) {
            if (e["type"] == type && (!final_only || e.value("final", false))) return &e;
        }
        return nullptr;
    };
    const json* ready = find("ready");
    ASSERT_NE(ready, nullptr);
    EXPECT_EQ((*ready)["input_rate"], demo.sample_rate);
    EXPECT_EQ((*ready)["output_rate"], 24000);
    EXPECT_EQ((*ready)["target_language"], "hi");
    EXPECT_EQ(find("error"), nullptr);
    const json* heard = find("transcript", true);
    ASSERT_NE(heard, nullptr);
    EXPECT_EQ((*heard)["text"], "I can't believe you did this!");
    const json* felt = find("emotion");
    ASSERT_NE(felt, nullptr);
    EXPECT_EQ((*felt)["label"], "anger");
    const json* translated = find("translation", true);
    ASSERT_NE(translated, nullptr);
    EXPECT_EQ((*translated)["text"], "मुझे यकीन नहीं हो रहा कि तुमने ऐसा किया!");
    EXPECT_NE(find("consistency"), nullptr);
    EXPECT_NE(find("playout"), nullptr);

    // "done" comes last, after the translated speech, with the session export.
    EXPECT_EQ(events.back()["type"], "done");
    ASSERT_EQ(events.back()["session"]["utterances"].size(), 1u);
    EXPECT_EQ(events.back()["session"]["utterances"][0]["translation"]["text"], (*translated)["text"]);
    EXPECT_GT(client.audio_bytes(), static_cast<std::size_t>(24000 * 2 / 2));  // over 0.5 s of speech
    s.server->stop();
    EXPECT_EQ(s.pool->active(), 0u);
}

// Over the wire: a warm session that has idled takes a client at another sample rate. A
// replacement loads after that client leaves, and the next client gets a fresh session.
TEST(WebSocketServer, ServesFromWarmSessionsAndNeverReusesOne) {
    const DemoInput demo = make_walkthrough_input();
    const Served s = start_server(write_script(demo, "warm"), 1, 1);
    ASSERT_NE(s.server, nullptr) << "no free port in 18765-18804";
    ASSERT_TRUE(eventually([&] { return s.pool->warm_ready() == 1; }, 60s));
    std::this_thread::sleep_for(300ms);  // the warm graph idles with no audio

    const auto run_client = [&](int rate) {
        Client client("ws://127.0.0.1:" + std::to_string(s.port) + "/?rate=" + std::to_string(rate));
        EXPECT_TRUE(client.wait_for("ready", 60s));
        client.send_pcm(Resampler::convert(demo.audio, demo.sample_rate, rate), rate);
        client.send_text(R"({"type": "end"})");
        EXPECT_TRUE(client.wait_for("done", 120s));
        return client.events();
    };

    const std::vector<json> first = run_client(48000);  // resampled to 16 kHz by the server
    ASSERT_FALSE(first.empty());
    EXPECT_EQ(first.front()["type"], "ready");
    EXPECT_EQ(first.front()["input_rate"], 48000);
    ASSERT_EQ(first.back()["type"], "done");
    ASSERT_EQ(first.back()["session"]["utterances"].size(), 1u);
    EXPECT_EQ(first.back()["session"]["utterances"][0]["source"]["text"], "I can't believe you did this!");

    // The replacement loads once the first client has gone (not during its session).
    ASSERT_TRUE(eventually([&] { return s.pool->active() == 0 && s.pool->warm_ready() == 1; }, 60s));
    const std::vector<json> second = run_client(demo.sample_rate);
    ASSERT_EQ(second.back()["type"], "done");
    EXPECT_EQ(second.back()["session"]["utterances"].size(), 1u);  // its own utterance only
    s.server->stop();
}

TEST(WebSocketServer, RefusesSessionsBeyondTheLimit) {
    const DemoInput demo = make_walkthrough_input();
    const Served s = start_server(write_script(demo, "limit"), 1);
    ASSERT_NE(s.server, nullptr) << "no free port in 18765-18804";
    const std::string url = "ws://127.0.0.1:" + std::to_string(s.port) + "/";

    Client first(url);
    ASSERT_TRUE(first.wait_for("ready", 60s));
    EXPECT_EQ(s.pool->active(), 1u);
    Client second(url);
    ASSERT_TRUE(second.wait_for("error", 30s));
    const std::vector<json> refused = second.events();
    EXPECT_NE(refused.front()["message"].get<std::string>().find("busy"), std::string::npos);
    EXPECT_EQ(s.pool->active(), 1u);
}

}  // namespace
}  // namespace ee
