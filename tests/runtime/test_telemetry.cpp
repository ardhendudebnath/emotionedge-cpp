#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <string>

#include "core/telemetry/ecs.hpp"
#include "core/telemetry/metrics.hpp"
#include "core/telemetry/telemetry.hpp"

namespace ee {
namespace {

using namespace std::chrono_literals;

TEST(Histogram, PercentilesWithinBucketPrecision) {
    telemetry::Histogram h;
    for (std::uint64_t v = 1; v <= 10'000; ++v) h.record(v);
    const auto s = h.snapshot();
    EXPECT_EQ(s.count, 10'000u);
    EXPECT_EQ(s.min, 1u);
    EXPECT_EQ(s.max, 10'000u);
    EXPECT_NEAR(static_cast<double>(s.percentile(0.5)), 5000.0, 5000.0 * 0.04);
    EXPECT_NEAR(static_cast<double>(s.percentile(0.95)), 9500.0, 9500.0 * 0.04);
    EXPECT_NEAR(s.mean(), 5000.5, 1e-6);
}

TEST(Histogram, SmallValuesAreExact) {
    telemetry::Histogram h;
    for (int i = 0; i < 10; ++i) h.record(7);
    EXPECT_EQ(h.snapshot().percentile(0.5), 7u);
    EXPECT_EQ(h.snapshot().percentile(0.99), 7u);
}

TEST(Histogram, BucketsCoverTheirValues) {
    for (std::uint64_t v : {0ull, 31ull, 32ull, 33ull, 1000ull, 123456ull, 1ull << 40}) {
        const auto i = telemetry::Histogram::bucket_index(v);
        EXPECT_LE(telemetry::Histogram::bucket_lower(i), v);
        EXPECT_GT(telemetry::Histogram::bucket_upper(i), v);
    }
}

TEST(Metrics, PrometheusExposition) {
    telemetry::MetricsRegistry reg;
    reg.counter("ee_frames_total", "Frames", {{"stage", "asr"}}).inc(3);
    reg.gauge("ee_depth", "Depth").set(2.5);
    auto& h = reg.histogram("ee_latency_seconds", "Latency", {{"stage", "mt"}});
    h.record(1000);  // 1 ms in µs
    const std::string text = reg.prometheus();
    EXPECT_NE(text.find("# TYPE ee_frames_total counter"), std::string::npos);
    EXPECT_NE(text.find("ee_frames_total{stage=\"asr\"} 3"), std::string::npos);
    EXPECT_NE(text.find("ee_depth 2.5"), std::string::npos);
    EXPECT_NE(text.find("# TYPE ee_latency_seconds summary"), std::string::npos);
    EXPECT_NE(text.find("ee_latency_seconds{stage=\"mt\",quantile=\"0.95\"} 0.001"), std::string::npos);
    EXPECT_NE(text.find("ee_latency_seconds_count{stage=\"mt\"} 1"), std::string::npos);
    // Same name + labels returns the same metric.
    reg.counter("ee_frames_total", "Frames", {{"stage", "asr"}}).inc();
    EXPECT_EQ(reg.find_counter("ee_frames_total", {{"stage", "asr"}})->value(), 4u);
}

TEST(EmotionConsistency, MatchesBlueprintFormula) {
    const Vad src{-0.62f, 0.78f, 0.55f};
    EXPECT_FLOAT_EQ(emotion_consistency(src, src), 1.0f);
    EXPECT_NEAR(emotion_consistency({-1, -1, -1}, {1, 1, 1}), 0.0f, 1e-6f);
    // Blueprint p.3: synthesized point slightly off the source.
    const Vad out{-0.53f, 0.68f, 0.55f};
    const float expected = 1.0f - std::sqrt(0.09f * 0.09f + 0.10f * 0.10f) / (2.0f * std::sqrt(3.0f));
    EXPECT_NEAR(emotion_consistency(src, out), expected, 1e-5f);
    EXPECT_GE(emotion_consistency(src, out), 0.75f);
}

TEST(Telemetry, FoldsMilestonesIntoLatencyBudgetRows) {
    telemetry::Telemetry t;
    using M = telemetry::Milestone;
    const TimePoint t0 = Clock::now();
    t.begin_utterance(1);
    t.mark(1, M::SpeechEnd, t0);
    t.mark(1, M::Endpoint, t0 + 160ms);
    t.mark(1, M::EmotionFinal, t0 + 190ms);
    t.mark(1, M::AsrFinal, t0 + 380ms);
    t.mark(1, M::StateReady, t0 + 390ms);
    t.mark(1, M::MtFinal, t0 + 510ms);
    t.mark(1, M::ControllerDone, t0 + 515ms);
    t.mark(1, M::TtsFirstChunk, t0 + 675ms);
    t.mark(1, M::TtsFirstChunk, t0 + 999ms);  // first mark wins
    EXPECT_EQ(t.completed_utterances(), 0u);
    t.mark(1, M::FirstAudio, t0 + 735ms);
    EXPECT_EQ(t.completed_utterances(), 1u);

    const auto rows = t.budget_report();
    ASSERT_EQ(rows.size(), 8u);
    const auto p50_ms = [&](std::size_t r) { return static_cast<double>(rows[r].snapshot.percentile(0.5)) / 1000.0; };
    EXPECT_NEAR(p50_ms(0), 160.0, 160 * 0.04);  // VAD endpoint hangover
    EXPECT_NEAR(p50_ms(1), 30.0, 30 * 0.04);    // emotion fusion (parallel)
    EXPECT_NEAR(p50_ms(2), 220.0, 220 * 0.04);  // ASR final decode
    EXPECT_NEAR(p50_ms(3), 10.0, 10 * 0.04);    // emotion state: from the later of ASR/emotion
    EXPECT_NEAR(p50_ms(6), 160.0, 160 * 0.04);  // TTS first chunk
    EXPECT_NEAR(static_cast<double>(t.end_to_end().percentile(0.5)) / 1000.0, 735.0, 735 * 0.04);

    // Marks for utterances that were never begun are ignored.
    t.mark(42, M::FirstAudio, t0);
    EXPECT_EQ(t.completed_utterances(), 1u);
    EXPECT_EQ(t.milestone_us(42, M::FirstAudio), -1);
}

TEST(Telemetry, BudgetTotalsSevenThirtyFiveMilliseconds) {
    std::uint64_t critical_path = 0;
    for (const auto& row : telemetry::latency_budget()) {
        if (row.on_critical_path) critical_path += row.target_us;
    }
    EXPECT_EQ(critical_path, 735'000u);  // 65 ms headroom under the 800 ms p95 target
    EXPECT_LT(critical_path, telemetry::kEndToEndTargetUs);
}

}  // namespace
}  // namespace ee
