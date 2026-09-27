#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#include "core/runtime/graph.hpp"
#include "core/runtime/stage_registry.hpp"
#include "core/telemetry/telemetry.hpp"

namespace ee {
namespace {

using namespace std::chrono_literals;

/// Emits `count` numbered frames, `burst` per tick, then finishes. With `alternate: true` it
/// interleaves audio (reliable) and partial transcripts (droppable).
class NumberSource final : public IStage {
public:
    void open(StageContext& ctx) override {
        ctx_ = &ctx;
        count_ = static_cast<int>(ctx.params().integer("count", 10));
        burst_ = static_cast<int>(ctx.params().integer("burst", 1));
        kind_ = parse_frame_kind(ctx.params().str("kind", "audio")).value();
        alternate_ = ctx.params().flag("alternate", false);
        final_last_ = ctx.params().flag("final_last", false);
    }
    void process(Frame&) override {}
    void tick() override {
        for (int b = 0; b < burst_ && next_ < count_; ++b, ++next_) {
            const FrameKind kind = alternate_ ? (next_ % 2 == 0 ? FrameKind::Audio : FrameKind::Transcript) : kind_;
            Frame& f = ctx_->make(kind);
            f.utterance = 1;
            f.seq = static_cast<std::uint32_t>(next_);
            if (final_last_ && next_ == count_ - 1) f.flags = frame_flags::kFinal;
            ctx_->emit(f);
        }
        if (next_ >= count_) ctx_->finish();
    }

private:
    StageContext* ctx_ = nullptr;
    int count_ = 0;
    int burst_ = 1;
    int next_ = 0;
    FrameKind kind_ = FrameKind::Audio;
    bool alternate_ = false;
    bool final_last_ = false;
};

/// Re-emits every frame with its sequence number doubled.
class Doubler final : public IStage {
public:
    void open(StageContext& ctx) override { ctx_ = &ctx; }
    void process(Frame& in) override {
        Frame& out = ctx_->make(in.kind);
        out.copy_header_from(in);
        out.flags = in.flags;
        out.seq = in.seq * 2;
        ctx_->emit(out);
    }

private:
    StageContext* ctx_ = nullptr;
};

/// Records what it receives; optionally slow, optionally throwing.
class Collector final : public IStage {
public:
    void open(StageContext& ctx) override {
        delay_ = std::chrono::microseconds(ctx.params().integer("delay_us", 0));
        throw_every_ = static_cast<int>(ctx.params().integer("throw_every", 0));
    }
    void process(Frame& f) override {
        if (delay_.count() > 0) std::this_thread::sleep_for(delay_);
        {
            std::lock_guard lock(mutex);
            frames.push_back({f.kind, f.seq});
        }
        if (throw_every_ > 0 && ++seen_ % throw_every_ == 0) throw std::runtime_error("boom");
    }
    void close() override { closed = true; }

    struct Seen {
        FrameKind kind;
        std::uint32_t seq;
    };
    std::vector<Seen> snapshot() {
        std::lock_guard lock(mutex);
        return frames;
    }
    std::mutex mutex;
    std::vector<Seen> frames;
    std::atomic<bool> closed{false};

private:
    std::chrono::microseconds delay_{0};
    int throw_every_ = 0;
    int seen_ = 0;
};

/// Forwards everything downstream and counts feedback frames (the 4.1 controller's role).
class LoopController final : public IStage {
public:
    void open(StageContext& ctx) override { ctx_ = &ctx; }
    void process(Frame& f) override {
        if (f.kind == FrameKind::Feedback) {
            ++feedback;
            return;
        }
        ctx_->emit(f);
    }
    std::atomic<int> feedback{0};

private:
    StageContext* ctx_ = nullptr;
};

/// Answers each input with a Feedback frame (the 5.2 consistency checker's role).
class FeedbackEcho final : public IStage {
public:
    void open(StageContext& ctx) override { ctx_ = &ctx; }
    void process(Frame& f) override {
        Frame& out = ctx_->make(FrameKind::Feedback);
        out.copy_header_from(f);
        out.score = 0.5f;
        ctx_->emit(out);
    }

private:
    StageContext* ctx_ = nullptr;
};

StageRegistry test_registry() {
    StageRegistry r;
    r.add<NumberSource>("numbers");
    r.add<Doubler>("doubler");
    r.add<Collector>("collector");
    r.add<LoopController>("loop_ctrl");
    r.add<FeedbackEcho>("feedback_echo");
    return r;
}

Collector& collector(Graph& g, std::string_view name = "sink") {
    auto* c = dynamic_cast<Collector*>(g.stage(name));
    EXPECT_NE(c, nullptr);
    return *c;
}

constexpr const char* kChain = R"(
threads:
  - { name: T1 }
  - { name: T2 }
stages:
  - { name: src, type: numbers, thread: T1, tick_ms: 1, params: { count: 100, burst: 7 } }
  - { name: x2, type: doubler, thread: T2 }
  - { name: sink, type: collector, thread: T2 }
edges:
  - { from: src, to: x2, capacity: 4 }
  - { from: x2, to: sink, capacity: 4 }
)";

void expect_doubled_sequence(Collector& sink) {
    const auto seen = sink.snapshot();
    ASSERT_EQ(seen.size(), 100u);
    for (std::size_t i = 0; i < seen.size(); ++i) EXPECT_EQ(seen[i].seq, 2 * i);
    EXPECT_TRUE(sink.closed);
}

TEST(Graph, DeterministicRunDeliversEverythingInOrder) {
    const auto registry = test_registry();
    Graph g(parse_pipeline(kChain), registry, {});
    ASSERT_TRUE(g.run_deterministic());
    EXPECT_TRUE(g.finished());
    expect_doubled_sequence(collector(g));
}

TEST(Graph, ThreadedRunMatchesDeterministicRun) {
    // x2 and sink share a thread and every edge holds only 4 frames: a blocking push from x2
    // into sink's full queue would deadlock, so this also guards the overflow path. Repeated,
    // because such hangs depend on timing.
    const auto registry = test_registry();
    for (int run = 0; run < 10; ++run) {
        Graph g(parse_pipeline(kChain), registry, {});
        g.start();
        ASSERT_TRUE(g.wait(10s)) << "run " << run << " did not finish";
        g.stop();
        expect_doubled_sequence(collector(g));
    }
}

TEST(Graph, CrossThreadCycleOfFullQueuesDoesNotDeadlock) {
    // A -> B -> A through two threads with tiny queues and bursts in both directions: with
    // blocking pushes each thread would wait on the other forever.
    const char* yaml = R"(
threads: [{ name: A }, { name: B }]
stages:
  - { name: src, type: numbers, thread: A, tick_ms: 1, params: { count: 400, burst: 50 } }
  - { name: relay, type: doubler, thread: B }
  - { name: back, type: doubler, thread: A }
  - { name: sink, type: collector, thread: B }
edges:
  - { from: src, to: relay, capacity: 2 }
  - { from: relay, to: back, capacity: 2 }
  - { from: back, to: sink, capacity: 2 }
)";
    const auto registry = test_registry();
    Graph g(parse_pipeline(yaml), registry, {});
    g.start();
    ASSERT_TRUE(g.wait(20s));
    g.stop();
    const auto seen = collector(g).snapshot();
    ASSERT_EQ(seen.size(), 400u);
    for (std::size_t i = 0; i < seen.size(); ++i) EXPECT_EQ(seen[i].seq, 4 * i);  // in order, none lost
}

TEST(Graph, FeedbackLoopDoesNotDeadlockEndOfStream) {
    const char* yaml = R"(
threads: [{ name: A }, { name: B }]
stages:
  - { name: src, type: numbers, thread: A, tick_ms: 1, params: { count: 20 } }
  - { name: ctrl, type: loop_ctrl, thread: A }
  - { name: tts, type: doubler, thread: B }
  - { name: ecs, type: feedback_echo, thread: B }
edges:
  - { from: src, to: ctrl }
  - { from: ctrl, to: tts }
  - { from: tts, to: ecs }
  - { from: ecs, to: ctrl, kinds: [feedback], feedback: true, capacity: 2 }
)";
    const auto registry = test_registry();
    for (bool threaded : {false, true}) {
        Graph g(parse_pipeline(yaml), registry, {});
        if (threaded) {
            g.start();
            ASSERT_TRUE(g.wait(10s)) << "threaded run did not finish";
            g.stop();
        } else {
            ASSERT_TRUE(g.run_deterministic());
        }
        auto* ctrl = dynamic_cast<LoopController*>(g.stage("ctrl"));
        ASSERT_NE(ctrl, nullptr);
        EXPECT_GT(ctrl->feedback.load(), 0);
    }
}

TEST(Graph, ConsumerSkipsSupersededPartials) {
    const char* yaml = R"(
stages:
  - { name: src, type: numbers, params: { count: 5, burst: 5, kind: transcript, final_last: true } }
  - { name: sink, type: collector }
edges:
  - { from: src, to: sink, capacity: 16 }
)";
    const auto registry = test_registry();
    Graph g(parse_pipeline(yaml), registry, {});
    ASSERT_TRUE(g.run_deterministic());
    const auto seen = collector(g).snapshot();
    ASSERT_EQ(seen.size(), 1u);  // four partials were superseded before the sink ran
    EXPECT_EQ(seen[0].seq, 4u);
    EXPECT_EQ(g.stale_frames(), 4u);
}

TEST(Graph, BackpressureShedsPartialsButNeverAudio) {
    const char* yaml = R"(
threads: [{ name: A }, { name: B }]
stages:
  - { name: src, type: numbers, thread: A, tick_ms: 1, params: { count: 200, burst: 200, alternate: true } }
  - { name: sink, type: collector, thread: B, params: { delay_us: 50 } }
edges:
  - { from: src, to: sink, capacity: 4 }
)";
    const auto registry = test_registry();
    Graph g(parse_pipeline(yaml), registry, {});
    g.start();
    ASSERT_TRUE(g.wait(20s));
    g.stop();

    std::size_t audio = 0;
    std::size_t transcripts = 0;
    std::uint32_t last_audio = 0;
    bool audio_in_order = true;
    for (const auto& s : collector(g).snapshot()) {
        if (s.kind == FrameKind::Audio) {
            audio_in_order = audio_in_order && (audio == 0 || s.seq > last_audio);
            last_audio = s.seq;
            ++audio;
        } else {
            ++transcripts;
        }
    }
    EXPECT_EQ(audio, 100u);
    EXPECT_TRUE(audio_in_order);
    EXPECT_EQ(transcripts + g.dropped_frames() + g.stale_frames(), 100u);
    EXPECT_GT(g.dropped_frames() + g.stale_frames(), 0u);
}

TEST(Graph, EdgesFilterByFrameKind) {
    const char* yaml = R"(
stages:
  - { name: src, type: numbers, params: { count: 10, alternate: true } }
  - { name: sink, type: collector }
edges:
  - { from: src, to: sink, kinds: [audio] }
)";
    const auto registry = test_registry();
    Graph g(parse_pipeline(yaml), registry, {});
    ASSERT_TRUE(g.run_deterministic());
    const auto seen = collector(g).snapshot();
    ASSERT_EQ(seen.size(), 5u);
    for (const auto& s : seen) EXPECT_EQ(s.kind, FrameKind::Audio);
}

TEST(Graph, StageExceptionsAreCountedNotFatal) {
    const char* yaml = R"(
stages:
  - { name: src, type: numbers, params: { count: 9 } }
  - { name: sink, type: collector, params: { throw_every: 3 } }
edges:
  - { from: src, to: sink }
)";
    const auto registry = test_registry();
    telemetry::Telemetry telemetry;
    Services services;
    services.telemetry = &telemetry;
    Graph g(parse_pipeline(yaml), registry, services);
    ASSERT_TRUE(g.run_deterministic());
    EXPECT_EQ(collector(g).snapshot().size(), 9u);
    const auto* errors = telemetry.metrics().find_counter("ee_stage_errors_total", {{"stage", "sink"}});
    ASSERT_NE(errors, nullptr);
    EXPECT_EQ(errors->value(), 3u);
}

TEST(Graph, UnknownStageTypeFailsAtConstruction) {
    const auto registry = test_registry();
    EXPECT_THROW(Graph(parse_pipeline("stages:\n  - { name: a, type: nope }\n"), registry, {}), ConfigError);
}

}  // namespace
}  // namespace ee
