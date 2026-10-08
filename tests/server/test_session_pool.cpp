// The live-session core both servers share (5.3): warm sessions, limits, resampling, and the
// order of what a client receives. A recording OutputChannel stands in for the transport, so
// these run in every build. Stand-in engines (config/pipeline.yaml).
#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "core/audio/resampler.hpp"
#include "core/pipeline/demo.hpp"
#include "core/server/session_pool.hpp"

namespace ee {
namespace {

namespace fs = std::filesystem;
using json = nlohmann::json;
using namespace std::chrono_literals;

/// What a transport would have sent, in order: "ready", the events, "audio" blocks, "done" or "error".
class RecordingChannel final : public OutputChannel {
public:
    void ready(const Ready& r) override {
        record({{"type", "ready"}, {"input_rate", r.input_rate}, {"output_rate", r.output_rate},
                {"target_language", r.target_language}});
    }
    void event(std::string_view text) override { record(json::parse(text)); }
    void audio(std::span<const std::int16_t> pcm) override {
        const std::lock_guard<std::mutex> lock(mu_);
        samples_ += pcm.size();
        if (kinds_.empty() || kinds_.back() != "audio") kinds_.emplace_back("audio");
    }
    void done(std::string_view session_json) override { record({{"type", "done"}, {"session", json::parse(session_json)}}); }
    void error(std::string_view message) override { record({{"type", "error"}, {"message", std::string(message)}}); }

    bool wait_for(const std::string& type, std::chrono::seconds timeout) {
        std::unique_lock<std::mutex> lock(mu_);
        return cv_.wait_for(lock, timeout, [&] {
            for (const json& e : events_) {
                if (e["type"] == type) return true;
            }
            return false;
        });
    }
    std::vector<json> events() {
        const std::lock_guard<std::mutex> lock(mu_);
        return events_;
    }
    std::vector<std::string> kinds() {
        const std::lock_guard<std::mutex> lock(mu_);
        return kinds_;
    }
    std::size_t samples() {
        const std::lock_guard<std::mutex> lock(mu_);
        return samples_;
    }

private:
    void record(json e) {
        const std::lock_guard<std::mutex> lock(mu_);
        kinds_.push_back(e["type"].get<std::string>());
        events_.push_back(std::move(e));
        cv_.notify_all();
    }

    std::mutex mu_;
    std::condition_variable cv_;
    std::vector<json> events_;
    std::vector<std::string> kinds_;  ///< event types and "audio" runs, in arrival order
    std::size_t samples_ = 0;
};

SessionPool::Options pool_options(const DemoInput& demo, const std::string& name, std::size_t max_sessions,
                                  std::size_t warm_sessions) {
    const fs::path dir = fs::temp_directory_path() / ("ee_pool_" + name);
    fs::create_directories(dir);
    std::ofstream(dir / "script.json", std::ios::binary) << demo.script_json;
    SessionPool::Options o;
    o.config = fs::path(EE_SOURCE_DIR) / "config" / "pipeline.yaml";
    o.max_sessions = max_sessions;
    o.warm_sessions = warm_sessions;
    // Long join timeouts: sanitizer builds must not take the "latest estimate" fallback.
    o.overrides = {{"asr.engine", "scripted"}, {"asr.script", (dir / "script.json").string()},
                   {"state.join_timeout_ms", "20000"}, {"emotion.transcript_timeout_ms", "20000"}};
    return o;
}

std::string pcm16(const std::vector<float>& audio) {
    std::string bytes;
    bytes.reserve(audio.size() * 2);
    for (const float x : audio) {
        const auto s = static_cast<std::uint16_t>(static_cast<std::int16_t>(std::clamp(x, -1.0f, 1.0f) * 32767.0f));
        bytes.push_back(static_cast<char>(s & 0xFFu));
        bytes.push_back(static_cast<char>(s >> 8));
    }
    return bytes;
}

/// Streams the speech in 20 ms blocks, ends it and waits for the stream to finish.
void speak(SessionStream& stream, const std::vector<float>& audio, int rate) {
    const std::string bytes = pcm16(audio);
    const std::size_t block = static_cast<std::size_t>(rate / 50) * 2;
    for (std::size_t pos = 0; pos < bytes.size(); pos += block) stream.audio(std::string_view(bytes).substr(pos, block));
    stream.end();
    stream.wait();
}

template <typename Pred>
bool eventually(Pred pred, std::chrono::seconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (!pred()) {
        if (std::chrono::steady_clock::now() > deadline) return false;
        std::this_thread::sleep_for(20ms);
    }
    return true;
}

TEST(SessionPool, StreamsReadyEventsSpeechThenDone) {
    const DemoInput demo = make_walkthrough_input();
    SessionPool pool(pool_options(demo, "stream", 1, 1));
    pool.start();
    RecordingChannel out;
    const auto stream = pool.open(out, demo.sample_rate);
    ASSERT_TRUE(out.wait_for("ready", 60s));
    speak(*stream, demo.audio, demo.sample_rate);
    pool.close(stream);

    const std::vector<std::string> kinds = out.kinds();
    ASSERT_FALSE(kinds.empty());
    EXPECT_EQ(kinds.front(), "ready");
    EXPECT_EQ(kinds.back(), "done");
    const auto first = [&](const std::string& k) { return std::find(kinds.begin(), kinds.end(), k) - kinds.begin(); };
    EXPECT_LT(first("transcript"), first("translation"));
    EXPECT_LT(first("translation"), first("audio"));
    EXPECT_GT(out.samples(), 24000u / 2);  // over 0.5 s of translated speech
    const std::vector<json> events = out.events();
    EXPECT_EQ(events.back()["session"]["utterances"].size(), 1u);
    EXPECT_EQ(events.back()["session"]["utterances"][0]["translation"]["text"], "मुझे यकीन नहीं हो रहा कि तुमने ऐसा किया!");
    EXPECT_EQ(pool.active(), 0u);
}

TEST(SessionPool, WarmSessionsServeOneClientEachAndResample) {
    const DemoInput demo = make_walkthrough_input();
    SessionPool pool(pool_options(demo, "warm", 1, 1));
    pool.start();
    ASSERT_TRUE(eventually([&] { return pool.warm_ready() == 1; }, 60s));
    std::this_thread::sleep_for(300ms);  // the warm graph idles with no audio

    RecordingChannel first;
    const auto a = pool.open(first, 48000);  // resampled to 16 kHz on the way in
    EXPECT_EQ(pool.warm_ready(), 0u);        // taken, and not replaced while it runs
    speak(*a, Resampler::convert(demo.audio, demo.sample_rate, 48000), 48000);
    pool.close(a);
    std::vector<json> events = first.events();
    EXPECT_EQ(events.front()["input_rate"], 48000);
    ASSERT_EQ(events.back()["type"], "done");
    EXPECT_EQ(events.back()["session"]["utterances"][0]["source"]["text"], "I can't believe you did this!");

    ASSERT_TRUE(eventually([&] { return pool.warm_ready() == 1; }, 60s));  // replaced once it ended
    RecordingChannel second;
    const auto b = pool.open(second, demo.sample_rate);
    speak(*b, demo.audio, demo.sample_rate);
    pool.close(b);
    events = second.events();
    ASSERT_EQ(events.back()["type"], "done");
    EXPECT_EQ(events.back()["session"]["utterances"].size(), 1u);  // its own utterance only
}

TEST(SessionPool, RefusesBeyondTheLimitAndBadRates) {
    const DemoInput demo = make_walkthrough_input();
    SessionPool pool(pool_options(demo, "limit", 1, 0));  // no warm session: loads on open
    pool.start();
    RecordingChannel a, b, c;
    const auto first = pool.open(a, 16000);
    try {
        (void)pool.open(b, 16000);
        FAIL() << "a second session beyond max_sessions";
    } catch (const SessionPool::Refused& r) {
        EXPECT_EQ(r.reason(), SessionPool::Refused::Reason::Busy);
    }
    try {
        (void)pool.open(c, 4000);
        FAIL() << "a 4 kHz client";
    } catch (const SessionPool::Refused& r) {
        EXPECT_EQ(r.reason(), SessionPool::Refused::Reason::BadRate);
    }
    EXPECT_EQ(pool.active(), 1u);
    pool.close(first);
    EXPECT_EQ(pool.active(), 0u);
    EXPECT_EQ(pool.warm_ready(), 0u);  // --warm 0: nothing kept loaded
}

TEST(SessionPool, StopEndsStreamsStillHeldByTransports) {
    const DemoInput demo = make_walkthrough_input();
    SessionPool pool(pool_options(demo, "stop", 1, 1));
    pool.start();
    RecordingChannel out;
    const auto stream = pool.open(out, demo.sample_rate);
    ASSERT_TRUE(out.wait_for("ready", 60s));
    pool.stop();
    stream->wait();  // returns: the stream was stopped
    pool.close(stream);  // and closing it afterwards is harmless
    EXPECT_EQ(pool.active(), 0u);
}

}  // namespace
}  // namespace ee
