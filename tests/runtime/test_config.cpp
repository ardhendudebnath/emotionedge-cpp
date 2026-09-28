#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <string>

#include "core/runtime/config.hpp"

namespace ee {
namespace {

constexpr const char* kPipeline = R"(
pipeline:
  name: test
  source_language: en
  target_language: es
  sample_rate: 16000
telemetry:
  flush_ms: 250
  prometheus: ${config_dir}/metrics.prom
threads:
  - { name: T1, core: 1, priority: high }
  - { name: T2 }
stages:
  - name: source
    type: frontend
    thread: T1
    tick_ms: 20
    params:
      agc: { enabled: true, target_dbfs: -23 }
      languages: [en, hi]
      model: ${config_dir}/model.onnx
  - name: middle
    type: pass
    thread: T2
  - name: sink
    type: pass
    thread: T2
edges:
  - { from: source, to: [middle, sink], kinds: [audio, control] }
  - { from: middle, to: sink, capacity: 8 }
  - { from: sink, to: middle, feedback: true, kinds: [feedback] }
)";

TEST(Config, ParsesThreadsStagesAndEdges) {
    const PipelineSpec spec = parse_pipeline(kPipeline, "/cfg");
    EXPECT_EQ(spec.name, "test");
    EXPECT_EQ(spec.target_language, "es");
    EXPECT_EQ(spec.telemetry.flush_ms, 250);
    EXPECT_EQ(spec.telemetry.prometheus_path, "/cfg/metrics.prom");

    ASSERT_EQ(spec.threads.size(), 2u);
    EXPECT_EQ(spec.threads[0].cores, std::vector<int>{1});
    EXPECT_TRUE(spec.threads[1].cores.empty());
    EXPECT_EQ(spec.threads[0].priority, ThreadPriority::High);
    EXPECT_EQ(spec.threads[1].priority, ThreadPriority::Normal);

    ASSERT_EQ(spec.stages.size(), 3u);
    const StageSpec& source = spec.stages[0];
    EXPECT_EQ(source.tick_ms, 20);
    EXPECT_TRUE(source.params.flag("agc.enabled", false));
    EXPECT_DOUBLE_EQ(source.params.number("agc.target_dbfs", 0), -23.0);
    EXPECT_EQ(source.params.list("languages"), (std::vector<std::string>{"en", "hi"}));
    EXPECT_EQ(source.params.str("model"), "/cfg/model.onnx");
    EXPECT_EQ(source.params.sub("agc").str("target_dbfs"), "-23");

    // "to: [middle, sink]" fans out into two edges.
    ASSERT_EQ(spec.edges.size(), 4u);
    EXPECT_EQ(spec.edges[0].to, "middle");
    EXPECT_EQ(spec.edges[1].to, "sink");
    EXPECT_EQ(spec.edges[0].kinds, kind_bit(FrameKind::Audio) | kind_bit(FrameKind::Control));
    EXPECT_EQ(spec.edges[2].capacity, 8u);
    EXPECT_EQ(spec.edges[2].kinds, kAllKinds);
    EXPECT_TRUE(spec.edges[3].feedback);
}

TEST(Config, TopologicalOrderIgnoresFeedbackEdges) {
    const PipelineSpec spec = parse_pipeline(kPipeline, "/cfg");
    const auto order = topological_order(spec);
    ASSERT_EQ(order.size(), 3u);
    const auto pos = [&](std::size_t stage) { return std::find(order.begin(), order.end(), stage) - order.begin(); };
    EXPECT_LT(pos(0), pos(1));
    EXPECT_LT(pos(1), pos(2));
}

TEST(Config, RejectsCyclesWithoutFeedbackEdge) {
    const char* yaml = R"(
stages:
  - { name: a, type: pass }
  - { name: b, type: pass }
edges:
  - { from: a, to: b }
  - { from: b, to: a }
)";
    EXPECT_THROW((void)parse_pipeline(yaml), ConfigError);
}

TEST(Config, DefaultsToOneThreadAndTypeEqualToName) {
    const PipelineSpec spec = parse_pipeline("stages:\n  - { name: asr }\n");
    ASSERT_EQ(spec.threads.size(), 1u);
    EXPECT_EQ(spec.stages[0].thread, spec.threads[0].name);
    EXPECT_EQ(spec.stages[0].type, "asr");
}

TEST(Config, PinsThreadsToCoreSets) {
    const PipelineSpec spec = parse_pipeline(R"(
threads:
  - { name: T2, cores: [2, 3, 4, 5] }
stages:
  - { name: asr, thread: T2, params: { threads: 4 } }
)");
    EXPECT_EQ(spec.threads[0].cores, (std::vector<int>{2, 3, 4, 5}));
    EXPECT_THROW((void)parse_pipeline("threads:\n  - { name: T, core: 1, cores: [2] }\nstages:\n  - { name: a, thread: T }\n"),
                 ConfigError);
    EXPECT_THROW((void)parse_pipeline("threads:\n  - { name: T, cores: [-1] }\nstages:\n  - { name: a, thread: T }\n"),
                 ConfigError);
}

// Engine worker pools inherit their thread's pinning: 4 whisper workers on 1 core stalled the
// real-time pipeline, so that configuration is rejected (also after a --set override).
TEST(Config, RejectsMoreWorkersThanPinnedCores) {
    const char* yaml = R"(
threads:
  - { name: T2, core: 2 }
  - { name: T3 }
stages:
  - { name: asr, thread: T2, params: { threads: 4 } }
  - { name: mt, thread: T3, params: { threads: 8 } }
)";
    EXPECT_THROW((void)parse_pipeline(yaml), ConfigError);
    PipelineSpec spec = parse_pipeline("threads:\n  - { name: T2, cores: [2, 3] }\nstages:\n"
                                       "  - { name: asr, thread: T2, params: { threads: 2 } }\n");
    EXPECT_NO_THROW(validate(spec));
    apply_override(spec, "asr.threads", "3");
    EXPECT_THROW(validate(spec), ConfigError);
}

TEST(Config, ReportsBrokenReferences) {
    EXPECT_THROW((void)parse_pipeline("stages:\n  - { name: a, thread: nope }\n"), ConfigError);
    EXPECT_THROW((void)parse_pipeline("stages:\n  - { name: a }\nedges:\n  - { from: a, to: b }\n"),
                 ConfigError);
    EXPECT_THROW((void)parse_pipeline("stages:\n  - { name: a }\n  - { name: a }\n"), ConfigError);
    EXPECT_THROW((void)parse_pipeline(
                     "stages:\n  - { name: a }\n  - { name: b }\nedges:\n  - { from: a, to: b, kinds: [bogus] }\n"),
                 ConfigError);
    EXPECT_THROW((void)parse_pipeline("not: [valid"), ConfigError);
}

TEST(Config, AppliesOverrides) {
    PipelineSpec spec = parse_pipeline(kPipeline, "/cfg");
    apply_override(spec, "pipeline.target_language", "hi");
    apply_override(spec, "middle.engine", "whisper");
    apply_override(spec, "middle.tick_ms", "10");
    apply_override(spec, "telemetry.trace", "trace.json");
    EXPECT_EQ(spec.target_language, "hi");
    EXPECT_EQ(spec.find_stage("middle")->params.str("engine"), "whisper");
    EXPECT_EQ(spec.find_stage("middle")->tick_ms, 10);
    EXPECT_EQ(spec.telemetry.trace_path, "trace.json");
    EXPECT_THROW(apply_override(spec, "nosuch.param", "1"), ConfigError);
    EXPECT_THROW(apply_override(spec, "noparam", "1"), ConfigError);
}

// pipeline.device is one switch for every model stage (phase 4): stages that set no device get
// it, a stage's own device wins, and --set pipeline.device lands before the graph applies it.
TEST(Config, PipelineDeviceIsTheStagesDefault) {
    PipelineSpec spec = parse_pipeline(R"(
pipeline: { device: auto }
stages:
  - { name: tts }
  - { name: vad, params: { device: cpu } }
)");
    EXPECT_EQ(spec.device, "auto");
    apply_override(spec, "pipeline.device", "cuda");
    spec.apply_device_default();
    EXPECT_EQ(spec.find_stage("tts")->params.str("device"), "cuda");
    EXPECT_EQ(spec.find_stage("vad")->params.str("device"), "cpu");

    PipelineSpec plain = parse_pipeline("stages:\n  - { name: tts }\n");
    plain.apply_device_default();
    EXPECT_FALSE(plain.find_stage("tts")->params.has("device"));
}

// Every graph shipped in config/ must parse, including the engine configs no test otherwise runs.
TEST(Config, ShippedPipelinesParse) {
    int parsed = 0;
    for (const auto& entry : std::filesystem::directory_iterator(std::filesystem::path(EE_SOURCE_DIR) / "config")) {
        const std::string name = entry.path().filename().string();
        if (name.rfind("pipeline", 0) != 0 || entry.path().extension() != ".yaml") continue;
        SCOPED_TRACE(name);
        PipelineSpec spec;
        EXPECT_NO_THROW(spec = load_pipeline(entry.path()));
        EXPECT_FALSE(spec.stages.empty());
        ++parsed;
    }
    EXPECT_GE(parsed, 2);
}

TEST(Params, TypedGettersValidateValues) {
    Params p;
    p.set("n", "2.5");
    p.set("i", "42");
    p.set("b", "yes");
    p.set("bad", "abc");
    EXPECT_DOUBLE_EQ(p.number("n", 0), 2.5);
    EXPECT_EQ(p.integer("i", 0), 42);
    EXPECT_TRUE(p.flag("b", false));
    EXPECT_EQ(p.integer("missing", 7), 7);
    EXPECT_THROW((void)p.number("bad", 0), ConfigError);
    EXPECT_THROW((void)p.integer("n", 0), ConfigError);
    EXPECT_THROW((void)p.flag("bad", false), ConfigError);
}

}  // namespace
}  // namespace ee
