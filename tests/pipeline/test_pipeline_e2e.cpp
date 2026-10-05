// End-to-end tests: the full config/pipeline.yaml graph on synthetic speech, plus "golden audio"
// features of the translated output (regenerate with EE_UPDATE_GOLDEN=1 after intended changes).
#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>

#include "core/audio/dsp.hpp"
#include "core/audio/pitch.hpp"
#include "core/audio/resampler.hpp"
#include "core/pipeline/demo.hpp"
#include "core/pipeline/session.hpp"
#include "core/translate/languages.hpp"

namespace ee {
namespace {

namespace fs = std::filesystem;
using json = nlohmann::json;

const std::string kHindi = "मुझे यकीन नहीं हो रहा कि तुमने ऐसा किया!";

fs::path fresh_dir(const std::string& name) {
    const fs::path dir = fs::temp_directory_path() / ("ee_e2e_" + name);
    fs::remove_all(dir);
    fs::create_directories(dir);
    return dir;
}

struct PipelineRun {
    SessionResult result;
    json session;
    std::string srt;
    double ecs_mean = 0.0;
    std::uint64_t end_to_end_count = 0;
};

/// Collects the live events (5.3) a session publishes.
class EventLog final : public IEventListener {
public:
    void on_event(std::string_view event) override {
        const std::lock_guard<std::mutex> lock(mu_);
        events_.push_back(json::parse(event));
    }
    [[nodiscard]] std::vector<json> events() const {
        const std::lock_guard<std::mutex> lock(mu_);
        return events_;
    }

private:
    mutable std::mutex mu_;
    std::vector<json> events_;
};

PipelineRun run_pipeline(const DemoInput& demo, const fs::path& dir, RunMode mode,
                 std::vector<std::pair<std::string, std::string>> extra = {}, IEventListener* events = nullptr) {
    std::ofstream(dir / "script.json", std::ios::binary) << demo.script_json;
    SessionOptions o;
    o.config = fs::path(EE_SOURCE_DIR) / "config" / "pipeline.yaml";
    o.mode = mode;
    o.events = events;
    o.speed = 4.0;
    o.input_samples = demo.audio;
    o.input_rate = demo.sample_rate;
    o.overrides = {{"asr.script", (dir / "script.json").string()},
                   {"recorder.srt", (dir / "captions.srt").string()},
                   {"recorder.json", (dir / "session.json").string()}};
    for (auto& kv : extra) o.overrides.push_back(std::move(kv));
    Session session(o);
    PipelineRun run;
    run.result = session.run();
    run.ecs_mean = session.telemetry().ecs().mean() / 10000.0;
    run.end_to_end_count = session.telemetry().end_to_end().count;
    std::ifstream srt(dir / "captions.srt", std::ios::binary);
    std::stringstream text;
    text << srt.rdbuf();
    run.srt = text.str();
    std::ifstream session_json(dir / "session.json");
    if (session_json) run.session = json::parse(session_json);
    return run;
}

struct AudioFeatures {
    double seconds = 0.0;
    double active_seconds = 0.0;
    double level_db = 0.0;
    double median_f0 = 0.0;
};

AudioFeatures features(const std::vector<float>& audio, int rate) {
    AudioFeatures f;
    f.seconds = static_cast<double>(audio.size()) / rate;
    const auto hop = static_cast<std::size_t>(rate / 100);
    std::vector<float> active;
    for (std::size_t pos = 0; pos + hop <= audio.size(); pos += hop) {
        const std::span<const float> frame(audio.data() + pos, hop);
        if (rms_dbfs(frame) > -45.0f) {
            f.active_seconds += 0.01;
            active.insert(active.end(), frame.begin(), frame.end());
        }
    }
    f.level_db = rms_dbfs(active);
    std::vector<float> f0;
    for (const PitchFrame& pf : track_pitch(Resampler::convert(audio, rate, 16000), PitchConfig{})) {
        if (pf.voiced) f0.push_back(pf.f0_hz);
    }
    std::sort(f0.begin(), f0.end());
    f.median_f0 = f0.empty() ? 0.0 : f0[f0.size() / 2];
    return f;
}

void check_golden(const std::string& name, const AudioFeatures& f) {
    const fs::path path = fs::path(EE_SOURCE_DIR) / "tests" / "golden" / (name + ".json");
    const json actual = {{"seconds", f.seconds}, {"active_seconds", f.active_seconds}, {"level_db", f.level_db},
                         {"median_f0", f.median_f0}};
    if (const char* update = std::getenv("EE_UPDATE_GOLDEN"); update != nullptr && std::string(update) == "1") {
        fs::create_directories(path.parent_path());
        std::ofstream(path) << actual.dump(2) << "\n";
        GTEST_SKIP() << "golden file written: " << path;
    }
    std::ifstream in(path);
    ASSERT_TRUE(in.good()) << "missing golden file " << path << " (run with EE_UPDATE_GOLDEN=1)";
    const json golden = json::parse(in);
    EXPECT_NEAR(f.seconds, golden["seconds"].get<double>(), 0.05);
    EXPECT_NEAR(f.active_seconds, golden["active_seconds"].get<double>(), 0.15);
    EXPECT_NEAR(f.level_db, golden["level_db"].get<double>(), 1.0);
    EXPECT_NEAR(f.median_f0, golden["median_f0"].get<double>(), 0.05 * golden["median_f0"].get<double>());
}

TEST(PipelineE2E, BlueprintWalkthroughOffline) {
    const fs::path dir = fresh_dir("walkthrough");
    const PipelineRun run = run_pipeline(make_walkthrough_input(), dir, RunMode::Offline);
    ASSERT_TRUE(run.result.completed);
    ASSERT_EQ(run.result.utterances.size(), 1u);
    const UtteranceRecord& u = run.result.utterances[0];

    // Blueprint p.3, step by step.
    EXPECT_EQ(u.source_text, "I can't believe you did this!");  // 1 · HEARD
    EXPECT_EQ(u.emotion.label, EmotionLabel::Anger);             // 2 · FELT
    EXPECT_LT(u.emotion.vad.v, 0.0f);
    EXPECT_GT(u.emotion.vad.a, 0.4f);
    EXPECT_GT(u.emotion.vad.d, 0.2f);
    EXPECT_EQ(u.source_emphasis, (std::vector<std::uint16_t>{2}));  // "believe"
    EXPECT_EQ(u.mt_input.rfind("<emo=anger a=", 0), 0u);            // 3 · MT INPUT
    EXPECT_NE(u.mt_input.find(" reg=casual> I can't <em>believe</em> you did this!"), std::string::npos);
    EXPECT_EQ(u.translation, kHindi);  // 4 · TRANSLATED
    ASSERT_EQ(u.target_emphasis.size(), 1u);
    EXPECT_EQ(split_words(u.translation, "hi")[u.target_emphasis[0]], "यकीन");
    EXPECT_GT(u.prosody.pitch_pct, 0.0f);  // 5 · PROSODY PLAN
    EXPECT_GT(u.prosody.range_pct, 0.0f);
    EXPECT_GT(u.prosody.rate_pct, 0.0f);
    EXPECT_GT(u.prosody.energy_db, 0.0f);
    EXPECT_GE(u.prosody.pause_ms, 60.0f);
    EXPECT_LE(u.prosody.pause_ms, 100.0f);
    EXPECT_GT(u.prosody.accent, 1.2f);
    ASSERT_FALSE(u.ecs.empty());  // 5.2 closed loop scored the output
    EXPECT_GE(u.ecs_mean(), 0.75f);
    EXPECT_GT(u.out_start, u.src_end);

    // 5.3 outputs.
    EXPECT_NE(run.srt.find("[anger] " + kHindi), std::string::npos);
    ASSERT_TRUE(run.session.contains("utterances"));
    EXPECT_EQ(run.session["utterances"][0]["translation"]["emphasis"][0], "यकीन");
    EXPECT_EQ(run.session["utterances"][0]["source"]["emphasis"][0], "believe");
    EXPECT_EQ(run.end_to_end_count, 1u);

    check_golden("walkthrough", features(run.result.output_audio, run.result.output_rate));
}

// 5.3 live outputs: each result goes out as an event while the session runs (what the
// WebSocket server streams), in pipeline order and tied to its utterance.
TEST(PipelineE2E, PublishesEachResultAsALiveEvent) {
    const fs::path dir = fresh_dir("events");
    EventLog log;
    const PipelineRun run = run_pipeline(make_walkthrough_input(), dir, RunMode::Offline, {}, &log);
    ASSERT_TRUE(run.result.completed);
    ASSERT_EQ(run.result.utterances.size(), 1u);
    const std::uint64_t id = run.result.utterances[0].id;

    const std::vector<json> events = log.events();
    const auto first = [&](const std::string& type, bool final_only = false) {
        for (std::size_t i = 0; i < events.size(); ++i) {
            if (events[i]["type"] == type && (!final_only || events[i].value("final", false))) return i;
        }
        return events.size();
    };
    const std::size_t heard = first("transcript", true), felt = first("emotion"), translated = first("translation", true),
                      planned = first("prosody"), scored = first("consistency"), played = first("playout");
    ASSERT_LT(heard, events.size());
    ASSERT_LT(felt, events.size());
    ASSERT_LT(translated, events.size());
    ASSERT_LT(planned, events.size());
    ASSERT_LT(scored, events.size());
    ASSERT_LT(played, events.size());
    EXPECT_LT(heard, translated);
    EXPECT_LT(translated, planned);
    EXPECT_LT(planned, played);

    EXPECT_EQ(events[heard]["text"], "I can't believe you did this!");
    EXPECT_EQ(events[felt]["label"], "anger");
    EXPECT_EQ(events[felt]["emphasis"][0], "believe");
    EXPECT_EQ(events[translated]["text"], kHindi);
    EXPECT_EQ(events[translated]["emphasis"][0], "यकीन");
    EXPECT_GT(events[planned]["pitch_pct"].get<double>(), 0.0);
    EXPECT_GT(events[scored]["ecs"].get<double>(), 0.0);
    EXPECT_LE(events[scored]["ecs"].get<double>(), 1.0);
    EXPECT_TRUE(events[scored]["heard"].contains("label"));
    EXPECT_GT(events[played]["end"].get<double>(), events[played]["start"].get<double>());
    for (const json& e : events) EXPECT_EQ(e["utterance"].get<std::uint64_t>(), id) << e.dump();
}

TEST(PipelineE2E, ConversationTracksEachEmotionAndClosesTheLoop) {
    const fs::path dir = fresh_dir("conversation");
    const PipelineRun run = run_pipeline(make_demo_conversation(), dir, RunMode::Offline);
    ASSERT_TRUE(run.result.completed);
    const auto& utterances = run.result.utterances;
    ASSERT_GE(utterances.size(), 3u);

    bool anger = false, sadness = false, joy = false;
    for (const UtteranceRecord& u : utterances) {
        anger = anger || u.emotion.label == EmotionLabel::Anger;
        sadness = sadness || u.emotion.label == EmotionLabel::Sadness;
        joy = joy || u.emotion.label == EmotionLabel::Joy;
        EXPECT_EQ(u.translation.rfind("[hi]", 0), std::string::npos) << "untranslated: " << u.source_text;
    }
    EXPECT_TRUE(anger);
    EXPECT_TRUE(sadness);
    EXPECT_TRUE(joy);
    EXPECT_GE(run.ecs_mean, 0.75);  // success target: ECS >= 0.75

    // Wherever a clause scored below the threshold, the controller corrected a later utterance.
    for (std::size_t i = 0; i < utterances.size(); ++i) {
        const auto& ecs = utterances[i].ecs;
        if (ecs.empty() || *std::min_element(ecs.begin(), ecs.end()) >= 0.75f) continue;
        bool corrected = false;
        for (std::size_t j = i + 1; j < utterances.size(); ++j) corrected = corrected || utterances[j].correction.norm() > 1e-4f;
        if (i + 1 < utterances.size()) {
            EXPECT_TRUE(corrected) << "no correction after utterance " << utterances[i].id;
        }
        break;
    }
}

TEST(PipelineE2E, RealtimeThreadedRunMatchesOffline) {
    // Functional equivalence of the threaded path, not a timing test: long join timeouts keep
    // slow machines (and sanitizer builds) from taking the degraded "latest estimate" fallback.
    const fs::path dir = fresh_dir("realtime");
    const PipelineRun run = run_pipeline(make_walkthrough_input(), dir, RunMode::Realtime,
                                         {{"state.join_timeout_ms", "20000"}, {"emotion.transcript_timeout_ms", "20000"}});
    ASSERT_TRUE(run.result.completed);
    ASSERT_EQ(run.result.utterances.size(), 1u);
    const UtteranceRecord& u = run.result.utterances[0];
    EXPECT_EQ(u.source_text, "I can't believe you did this!");
    EXPECT_EQ(u.translation, kHindi);
    EXPECT_EQ(u.emotion.label, EmotionLabel::Anger);
    EXPECT_EQ(run.end_to_end_count, 1u);
    EXPECT_FALSE(run.result.output_audio.empty());
}

TEST(PipelineE2E, OtherTargetLanguages) {
    const fs::path dir = fresh_dir("spanish");
    const PipelineRun run = run_pipeline(make_walkthrough_input(), dir, RunMode::Offline, {{"pipeline.target_language", "es"}});
    ASSERT_EQ(run.result.utterances.size(), 1u);
    const UtteranceRecord& u = run.result.utterances[0];
    EXPECT_EQ(u.translation, "¡No puedo creer que hayas hecho esto!");
    ASSERT_EQ(u.target_emphasis.size(), 1u);
    EXPECT_EQ(split_words(u.translation, "es")[u.target_emphasis[0]], "creer");
}

}  // namespace
}  // namespace ee
