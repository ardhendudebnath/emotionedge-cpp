#include "core/runtime/graph.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <exception>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "core/runtime/log.hpp"
#include "core/runtime/spsc_queue.hpp"
#include "core/runtime/thread_util.hpp"
#include "core/telemetry/telemetry.hpp"
#include "core/telemetry/trace.hpp"

namespace ee {

namespace {

/// Adaptive idle wait: yield a few times, then sleep in short slices (never past `deadline`).
class Backoff {
public:
    void reset() noexcept { spins_ = 0; }
    void idle(TimePoint deadline) {
        if (spins_ < 32) {
            ++spins_;
            std::this_thread::yield();
            return;
        }
        const auto now = Clock::now();
        const auto slice = std::chrono::microseconds(500);
        if (deadline <= now) return;
        std::this_thread::sleep_for(std::min<Clock::duration>(slice, deadline - now));
    }

private:
    int spins_ = 0;
};

constexpr std::size_t kTraceRingCapacity = 8192;
constexpr std::size_t kMaxFramesPerStep = 64;

}  // namespace

struct Graph::Impl {
    struct Node;

    /// An edge is a lock-free SPSC queue (the fast path) plus an unbounded spill for reliable
    /// frames that find it full, so a full queue can never deadlock the scheduler: not between
    /// two stages on one thread, and not around a cycle of threads (T3 -> T5 -> T3).
    struct Edge {
        Edge(const EdgeSpec& s, std::size_t f, std::size_t t) : spec(s), from(f), to(t), queue(s.capacity) {}
        EdgeSpec spec;
        std::size_t from;
        std::size_t to;
        bool same_thread = false;
        SpscQueue<Frame> queue;
        // Producer and consumer on one thread (deterministic runs, or co-scheduled stages):
        // a plain deque, touched by that thread only.
        std::deque<Frame> overflow;
        // Producer and consumer on different threads: a mutex-guarded spill, used only on the
        // slow path. The consumer moves one spilled frame at a time into `held`.
        std::mutex spill_mutex;
        std::deque<Frame> spill;
        std::atomic<std::size_t> spill_size{0};
        Frame held;
        bool holding = false;
        telemetry::Counter* dropped = nullptr;
        telemetry::Counter* stale = nullptr;
        telemetry::Counter* stalls = nullptr;
        telemetry::Counter* spills = nullptr;
        telemetry::Gauge* depth = nullptr;
    };

    class NodeContext final : public StageContext {
    public:
        NodeContext(Impl& graph, Node& node) : graph_(graph), node_(node) {}
        std::string_view name() const noexcept override { return node_.spec->name; }
        const Params& params() const noexcept override { return node_.spec->params; }
        const PipelineSpec& pipeline() const noexcept override { return graph_.spec; }
        Services& services() noexcept override { return graph_.services; }
        Frame& make(FrameKind kind) override {
            scratch_.reset(kind);
            return scratch_;
        }
        void emit(const Frame& frame) override { graph_.emit(node_, frame); }
        void finish() override { node_.finish_requested = true; }
        bool deterministic() const noexcept override { return graph_.mode == ExecutionMode::Deterministic; }

    private:
        Impl& graph_;
        Node& node_;
        Frame scratch_;
    };

    struct Node {
        std::size_t index = 0;
        const StageSpec* spec = nullptr;
        std::unique_ptr<IStage> stage;
        std::unique_ptr<NodeContext> ctx;
        std::vector<Edge*> inputs;
        std::vector<Edge*> outputs;
        std::size_t eos_needed = 0;  // non-feedback inputs
        std::size_t eos_seen = 0;
        bool finish_requested = false;
        std::atomic<bool> closed{false};
        TimePoint next_tick{};
        std::uint32_t tid = 0;
        telemetry::Histogram* process_us = nullptr;
        telemetry::Counter* frames_in = nullptr;
        telemetry::Counter* frames_out = nullptr;
        telemetry::Counter* errors = nullptr;
        std::unique_ptr<SpscQueue<telemetry::TraceEvent>> trace;
        const char* trace_name = "";
    };

    PipelineSpec spec;
    Services services;
    std::unique_ptr<telemetry::Telemetry> own_telemetry;
    std::vector<std::unique_ptr<Node>> nodes;
    std::vector<std::unique_ptr<Edge>> edges;
    std::vector<std::size_t> order;
    ExecutionMode mode = ExecutionMode::Deterministic;
    bool opened = false;

    std::atomic<bool> stop_requested{false};
    std::atomic<std::size_t> closed_count{0};
    std::atomic<std::uint64_t> activity{0};
    std::vector<std::thread> threads;
    std::thread telemetry_thread;
    std::mutex done_mutex;
    std::condition_variable done_cv;
    std::mutex flush_mutex;

    std::unique_ptr<telemetry::TraceWriter> trace_writer;
    TimePoint trace_epoch{};

    // ---- construction ------------------------------------------------------------------------

    Impl(PipelineSpec s, const StageRegistry& registry, Services svc) : spec(std::move(s)), services(svc) {
        validate(spec);
        if (services.telemetry == nullptr) {
            own_telemetry = std::make_unique<telemetry::Telemetry>();
            services.telemetry = own_telemetry.get();
        }
        auto& metrics = services.telemetry->metrics();

        std::unordered_map<std::string, std::size_t> index;
        std::unordered_map<std::string, std::uint32_t> thread_ids;
        for (std::size_t i = 0; i < spec.threads.size(); ++i) {
            thread_ids[spec.threads[i].name] = static_cast<std::uint32_t>(i + 1);
        }
        for (std::size_t i = 0; i < spec.stages.size(); ++i) {
            const StageSpec& ss = spec.stages[i];
            auto node = std::make_unique<Node>();
            node->index = i;
            node->spec = &spec.stages[i];
            node->stage = registry.create(ss.type);
            node->ctx = std::make_unique<NodeContext>(*this, *node);
            node->tid = thread_ids[ss.thread];
            node->trace_name = ss.name.c_str();
            const telemetry::Labels labels = {{"stage", ss.name}};
            node->process_us = &metrics.histogram("ee_stage_process_seconds",
                                                  "Time spent in IStage::process per frame", labels);
            node->frames_in = &metrics.counter("ee_stage_frames_in_total", "Frames processed", labels);
            node->frames_out = &metrics.counter("ee_stage_frames_out_total", "Frames emitted", labels);
            node->errors = &metrics.counter("ee_stage_errors_total", "Exceptions caught in a stage", labels);
            index[ss.name] = i;
            nodes.push_back(std::move(node));
        }
        for (const EdgeSpec& es : spec.edges) {
            const std::size_t from = index.at(es.from);
            const std::size_t to = index.at(es.to);
            auto edge = std::make_unique<Edge>(es, from, to);
            const telemetry::Labels labels = {{"from", es.from}, {"to", es.to}};
            edge->dropped = &metrics.counter("ee_edge_dropped_total",
                                             "Droppable frames shed on a full queue", labels);
            edge->stale = &metrics.counter("ee_edge_stale_total",
                                           "Partials skipped because a newer one was queued", labels);
            edge->stalls = &metrics.counter("ee_edge_backpressure_total",
                                             "Times a producer found a full queue for a reliable frame", labels);
            edge->spills = &metrics.counter("ee_edge_spilled_total",
                                            "Reliable frames parked in the overflow of a full queue", labels);
            edge->depth = &metrics.gauge("ee_edge_queue_depth", "Frames waiting in the queue", labels);
            edge->same_thread = nodes[from]->spec->thread == nodes[to]->spec->thread;
            // Pre-size audio buffers in every slot so steady-state audio traffic never allocates.
            if ((es.kinds & (kind_bit(FrameKind::Audio) | kind_bit(FrameKind::SynthAudio))) != 0) {
                edge->queue.for_each_slot([](Frame& f) { f.audio.reserve(2400); });
            }
            nodes[from]->outputs.push_back(edge.get());
            nodes[to]->inputs.push_back(edge.get());
            if (!es.feedback) ++nodes[to]->eos_needed;
            edges.push_back(std::move(edge));
        }
        order = topological_order(spec);
    }

    // ---- queue access -----------------------------------------------------------------------

    /// True when producer and consumer of `e` run on the same thread.
    [[nodiscard]] bool local(const Edge& e) const noexcept {
        return mode == ExecutionMode::Deterministic || e.same_thread;
    }

    static bool try_fast(Edge& e, const Frame& f) {
        Frame* slot = e.queue.acquire();
        if (slot == nullptr) return false;
        *slot = f;
        e.queue.publish();
        return true;
    }

    bool push(Edge& e, const Frame& f) {
        Node& consumer = *nodes[e.to];
        if (consumer.closed.load(std::memory_order_acquire)) return false;
        const bool shed = mode == ExecutionMode::Threaded && (e.spec.feedback || is_droppable(f));

        if (local(e)) {
            // FIFO: while anything is parked, new frames queue up behind it.
            if (e.overflow.empty() && try_fast(e, f)) return true;
            if (shed) {
                e.dropped->inc();
                return false;
            }
            if (mode == ExecutionMode::Threaded) e.stalls->inc();
            e.overflow.push_back(f);
            return true;
        }

        if (e.spill_size.load(std::memory_order_acquire) == 0 && try_fast(e, f)) return true;
        if (shed) {  // "drops stale partials, never audio"
            e.dropped->inc();
            return false;
        }
        e.stalls->inc();
        // Backpressure: give the consumer a moment before parking the frame.
        if (e.spill_size.load(std::memory_order_acquire) == 0) {
            const TimePoint until = Clock::now() + std::chrono::milliseconds(2);
            Backoff backoff;
            while (Clock::now() < until && !stop_requested.load(std::memory_order_relaxed) &&
                   !consumer.closed.load(std::memory_order_acquire)) {
                if (try_fast(e, f)) return true;
                backoff.idle(until);
            }
        }
        std::lock_guard lock(e.spill_mutex);
        e.spill.push_back(f);
        e.spill_size.fetch_add(1, std::memory_order_release);
        e.spills->inc();
        return true;
    }

    /// Frame `i` places behind the front (i = 0 or 1), in FIFO order across queue and overflow.
    Frame* peek(Edge& e, std::size_t i) {
        if (local(e)) {
            if (Frame* f = e.queue.peek(i)) return f;
            if (e.overflow.empty()) return nullptr;
            const std::size_t queued = e.queue.size();  // exact: this thread is both ends
            if (i >= queued && i - queued < e.overflow.size()) return &e.overflow[i - queued];
            return nullptr;
        }
        // A frame taken out of the spill is older than anything queued after it.
        if (e.holding) return i == 0 ? &e.held : nullptr;
        if (Frame* f = e.queue.peek(i)) return f;
        if (i > 0 || e.spill_size.load(std::memory_order_acquire) == 0) return nullptr;
        std::lock_guard lock(e.spill_mutex);
        if (e.spill.empty()) return nullptr;
        using std::swap;
        swap(e.held, e.spill.front());
        e.spill.pop_front();
        e.spill_size.fetch_sub(1, std::memory_order_release);
        e.holding = true;
        return &e.held;
    }

    void pop(Edge& e) {
        if (!local(e) && e.holding) {
            e.holding = false;
        } else if (e.queue.front() != nullptr) {
            e.queue.pop();
        } else if (!e.overflow.empty()) {
            e.overflow.pop_front();
        }
    }

    // ---- stage execution --------------------------------------------------------------------

    void emit(Node& node, const Frame& f) {
        for (Edge* e : node.outputs) {
            if ((e->spec.kinds & kind_bit(f.kind)) != 0) push(*e, f);
        }
        node.frames_out->inc();
        activity.fetch_add(1, std::memory_order_relaxed);
    }

    void deliver(Node& node, Frame& f) {
        const TimePoint t0 = Clock::now();
        try {
            node.stage->process(f);
        } catch (const std::exception& ex) {
            node.errors->inc();
            log::error("stage '", node.spec->name, "' failed on a ", to_string(f.kind), " frame: ", ex.what());
        }
        const TimePoint t1 = Clock::now();
        node.process_us->record(static_cast<std::uint64_t>(micros_between(t0, t1)));
        node.frames_in->inc();
        if (node.trace) {
            telemetry::TraceEvent ev;
            ev.name = node.trace_name;
            ev.category = to_string(f.kind).data();
            ev.ts_us = micros_between(trace_epoch, t0);
            ev.dur_us = micros_between(t0, t1);
            ev.tid = node.tid;
            ev.utterance = f.utterance;
            (void)node.trace->try_push(ev);
        }
        activity.fetch_add(1, std::memory_order_relaxed);
    }

    void run_tick(Node& node) {
        try {
            node.stage->tick();
        } catch (const std::exception& ex) {
            node.errors->inc();
            log::error("stage '", node.spec->name, "' tick failed: ", ex.what());
        }
    }

    void close_node(Node& node) {
        try {
            node.stage->close();
        } catch (const std::exception& ex) {
            node.errors->inc();
            log::error("stage '", node.spec->name, "' close failed: ", ex.what());
        }
        Frame eos;
        eos.reset(FrameKind::Control);
        eos.flags = frame_flags::kEndOfStream;
        for (Edge* e : node.outputs) push(*e, eos);
        node.closed.store(true, std::memory_order_release);
        activity.fetch_add(1, std::memory_order_relaxed);
        if (closed_count.fetch_add(1, std::memory_order_acq_rel) + 1 == nodes.size()) {
            std::lock_guard lock(done_mutex);
            done_cv.notify_all();
        }
    }

    /// One scheduling pass over a node: drain its inputs, tick it if due, close it if done.
    bool step(Node& node, TimePoint now) {
        if (node.closed.load(std::memory_order_relaxed)) return false;
        bool worked = false;
        for (Edge* e : node.inputs) {
            for (std::size_t n = 0; n < kMaxFramesPerStep; ++n) {
                Frame* f = peek(*e, 0);
                if (f == nullptr) break;
                worked = true;
                if (f->is_end_of_stream()) {
                    if (!e->spec.feedback) ++node.eos_seen;
                    pop(*e);
                    continue;
                }
                if (is_droppable(*f)) {
                    const Frame* next = peek(*e, 1);
                    if (next != nullptr && next->kind == f->kind && next->utterance == f->utterance) {
                        e->stale->inc();
                        pop(*e);
                        continue;
                    }
                }
                deliver(node, *f);
                pop(*e);
            }
        }

        const bool is_source = node.inputs.empty();
        if (node.spec->tick_ms > 0 || is_source) {
            if (mode == ExecutionMode::Deterministic) {
                run_tick(node);
            } else if (now >= node.next_tick) {
                run_tick(node);
                const auto period = std::chrono::milliseconds(std::max(node.spec->tick_ms, 1));
                node.next_tick += period;
                if (node.next_tick < now - 4 * period) node.next_tick = now + period;  // no bursts after a stall
            }
        }

        const bool inputs_done = node.eos_needed > 0 && node.eos_seen >= node.eos_needed;
        if (inputs_done || node.finish_requested) {
            close_node(node);
            worked = true;
        }
        return worked;
    }

    void open_all() {
        if (opened) return;
        opened = true;
        if (!spec.telemetry.trace_path.empty()) {
            trace_writer = std::make_unique<telemetry::TraceWriter>(spec.telemetry.trace_path);
            if (!trace_writer->ok()) {
                log::warn("cannot write trace to '", spec.telemetry.trace_path, "'");
                trace_writer.reset();
            } else {
                trace_epoch = Clock::now();
                for (std::size_t i = 0; i < spec.threads.size(); ++i) {
                    trace_writer->thread_name(static_cast<std::uint32_t>(i + 1), spec.threads[i].name);
                }
                for (auto& node : nodes) {
                    node->trace = std::make_unique<SpscQueue<telemetry::TraceEvent>>(kTraceRingCapacity);
                }
            }
        }
        for (std::size_t idx : order) {
            Node& node = *nodes[idx];
            node.stage->open(*node.ctx);
        }
    }

    // ---- telemetry (T7) ---------------------------------------------------------------------

    void drain_traces() {
        if (!trace_writer) return;
        telemetry::TraceEvent ev;
        for (auto& node : nodes) {
            while (node->trace && node->trace->try_pop(ev)) trace_writer->write(ev);
        }
    }

    void flush_telemetry() {
        std::lock_guard lock(flush_mutex);
        for (auto& e : edges) {
            // The local overflow belongs to its thread; only deterministic runs may read it here.
            const std::size_t local_backlog = mode == ExecutionMode::Deterministic ? e->overflow.size() : 0;
            e->depth->set(static_cast<double>(e->queue.size() + local_backlog + e->spill_size.load()));
        }
        drain_traces();
        if (!spec.telemetry.prometheus_path.empty()) {
            if (!telemetry::write_file_atomically(spec.telemetry.prometheus_path,
                                                  services.telemetry->metrics().prometheus())) {
                log::warn("cannot write metrics to '", spec.telemetry.prometheus_path, "'");
            }
        }
    }

    void telemetry_main() {
        set_current_thread_name("T7-telemetry");
        set_current_thread_priority(ThreadPriority::Low);
        const auto period = std::chrono::milliseconds(std::max(spec.telemetry.flush_ms, 10));
        std::unique_lock lock(done_mutex);
        while (!stop_requested.load(std::memory_order_acquire) && closed_count.load() < nodes.size()) {
            done_cv.wait_for(lock, period);
            lock.unlock();
            flush_telemetry();
            lock.lock();
        }
    }

    void thread_main(std::size_t thread_index) {
        const ThreadSpec& ts = spec.threads[thread_index];
        set_current_thread_name(ts.name);
        if (ts.core >= 0 && !pin_current_thread(ts.core)) {
            log::debug("thread ", ts.name, ": could not pin to core ", ts.core);
        }
        if (ts.priority != ThreadPriority::Normal && !set_current_thread_priority(ts.priority)) {
            log::debug("thread ", ts.name, ": priority request not granted");
        }
        std::vector<Node*> mine;
        for (std::size_t idx : order) {
            if (nodes[idx]->spec->thread == ts.name) mine.push_back(nodes[idx].get());
        }
        const TimePoint start = Clock::now();
        for (Node* n : mine) n->next_tick = start;

        Backoff backoff;
        while (!stop_requested.load(std::memory_order_acquire)) {
            const TimePoint now = Clock::now();
            TimePoint deadline = now + std::chrono::milliseconds(2);
            bool worked = false;
            bool all_closed = true;
            for (Node* n : mine) {
                if (n->closed.load(std::memory_order_relaxed)) continue;
                all_closed = false;
                worked |= step(*n, now);
                if (n->spec->tick_ms > 0) deadline = std::min(deadline, n->next_tick);
            }
            if (all_closed) break;
            if (worked) {
                backoff.reset();
            } else {
                backoff.idle(deadline);
            }
        }
    }
};

Graph::Graph(PipelineSpec spec, const StageRegistry& registry, Services services)
    : impl_(std::make_unique<Impl>(std::move(spec), registry, services)) {}

Graph::~Graph() { stop(); }

bool Graph::run_deterministic() {
    Impl& g = *impl_;
    g.mode = ExecutionMode::Deterministic;
    g.open_all();
    int idle_passes = 0;
    bool ok = true;
    while (g.closed_count.load() < g.nodes.size()) {
        const std::uint64_t before = g.activity.load(std::memory_order_relaxed);
        for (std::size_t idx : g.order) g.step(*g.nodes[idx], Clock::now());
        g.drain_traces();
        if (g.activity.load(std::memory_order_relaxed) == before) {
            if (++idle_passes >= 3) {
                std::string open;
                for (auto& n : g.nodes) {
                    if (!n->closed) open += (open.empty() ? "" : ", ") + n->spec->name;
                }
                log::error("pipeline stalled; stages still open: ", open);
                ok = false;
                break;
            }
        } else {
            idle_passes = 0;
        }
    }
    g.flush_telemetry();
    if (g.trace_writer) g.trace_writer->close();
    return ok;
}

void Graph::start() {
    Impl& g = *impl_;
    if (!g.threads.empty()) return;
    g.mode = ExecutionMode::Threaded;
    g.open_all();
    for (std::size_t i = 0; i < g.spec.threads.size(); ++i) {
        g.threads.emplace_back([&g, i] { g.thread_main(i); });
    }
    const bool wants_flush = !g.spec.telemetry.prometheus_path.empty() || g.trace_writer != nullptr;
    if (wants_flush && g.spec.telemetry.flush_ms > 0) {
        g.telemetry_thread = std::thread([&g] { g.telemetry_main(); });
    }
}

bool Graph::wait(std::chrono::milliseconds timeout) {
    Impl& g = *impl_;
    std::unique_lock lock(g.done_mutex);
    const auto done = [&g] {
        return g.closed_count.load() >= g.nodes.size() || g.stop_requested.load();
    };
    if (timeout == std::chrono::milliseconds::max()) {
        g.done_cv.wait(lock, done);
    } else if (!g.done_cv.wait_for(lock, timeout, done)) {
        return false;
    }
    return g.closed_count.load() >= g.nodes.size();
}

void Graph::stop() {
    Impl& g = *impl_;
    {
        std::lock_guard lock(g.done_mutex);
        g.stop_requested.store(true, std::memory_order_release);
        g.done_cv.notify_all();
    }
    for (std::thread& t : g.threads) {
        if (t.joinable()) t.join();
    }
    g.threads.clear();
    if (g.telemetry_thread.joinable()) g.telemetry_thread.join();
    if (g.opened && g.mode == ExecutionMode::Threaded) {
        g.flush_telemetry();
        if (g.trace_writer) g.trace_writer->close();
    }
}

bool Graph::finished() const noexcept { return impl_->closed_count.load() >= impl_->nodes.size(); }

const PipelineSpec& Graph::spec() const noexcept { return impl_->spec; }

Services& Graph::services() noexcept { return impl_->services; }

IStage* Graph::stage(std::string_view name) noexcept {
    for (auto& n : impl_->nodes) {
        if (n->spec->name == name) return n->stage.get();
    }
    return nullptr;
}

std::uint64_t Graph::dropped_frames() const noexcept {
    std::uint64_t total = 0;
    for (const auto& e : impl_->edges) total += e->dropped->value();
    return total;
}

std::uint64_t Graph::stale_frames() const noexcept {
    std::uint64_t total = 0;
    for (const auto& e : impl_->edges) total += e->stale->value();
    return total;
}

}  // namespace ee
