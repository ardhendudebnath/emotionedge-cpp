#include "core/server/session_pool.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>

#include "core/audio/audio_io.hpp"
#include "core/audio/resampler.hpp"
#include "core/pipeline/recorder.hpp"
#include "core/pipeline/session.hpp"
#include "core/runtime/log.hpp"

namespace ee {

// Named, not anonymous: SessionStream::Impl and SessionPool::Impl hold these.
namespace server_detail {

/// Every session takes speech at this rate; streams resample other rates on the way in.
constexpr int kSessionRate = 16000;

/// A live pipeline session with its own audio endpoints. It is built, models and all, on its own
/// thread as soon as it is created. Until a stream takes it, it idles: ready, with no audio.
class LiveSession final : public IEventListener {
public:
    explicit LiveSession(const SessionPool::Options& options)
        : source_(kSessionRate, static_cast<std::size_t>(kSessionRate) * 30),
          sink_(options.output_rate, static_cast<std::size_t>(options.output_rate) * 60) {
        io_.capture = &source_;
        io_.playback = &sink_;
        SessionOptions so;
        so.config = options.config;
        so.overrides = options.overrides;
        so.mode = RunMode::Live;
        so.live_io = &io_;
        so.events = this;
        so.output_rate = options.output_rate;
        so.on_ready = [this] { ready_.store(true, std::memory_order_release); };
        session_ = std::make_unique<Session>(std::move(so));  // a bad config throws here
        runner_ = std::thread([this] { run(); });
    }
    ~LiveSession() override { stop(); }
    LiveSession(const LiveSession&) = delete;
    LiveSession& operator=(const LiveSession&) = delete;

    /// Events go to the stream that took the session (none come before its audio does).
    void on_event(std::string_view event) override {
        if (IEventListener* target = target_.load(std::memory_order_acquire)) target->on_event(event);
    }
    void attach(IEventListener* target) noexcept { target_.store(target, std::memory_order_release); }

    [[nodiscard]] bool ready() const noexcept { return ready_.load(std::memory_order_acquire); }
    /// The session has ended: done() or error() says how.
    [[nodiscard]] bool finished() const noexcept { return finished_.load(std::memory_order_acquire); }
    /// Valid once finished().
    [[nodiscard]] const std::string& done() const noexcept { return done_; }
    [[nodiscard]] const std::string& error() const noexcept { return error_; }
    [[nodiscard]] const PipelineSpec& spec() const noexcept { return session_->spec(); }
    [[nodiscard]] RingSource& source() noexcept { return source_; }
    [[nodiscard]] RingSink& sink() noexcept { return sink_; }

    /// Ends the session now and joins its thread. Idempotent.
    void stop() {
        const std::lock_guard<std::mutex> lock(stop_mu_);
        source_.close();
        session_->stop();
        if (runner_.joinable()) runner_.join();
    }

private:
    void run() {
        try {
            const SessionResult result = session_->run();
            done_ = to_session_json(result.utterances, session_->spec(), &session_->telemetry());
        } catch (const std::exception& e) {
            log::error("server: session failed: ", e.what());
            error_ = e.what();
        }
        finished_.store(true, std::memory_order_release);
    }

    RingSource source_;
    RingSink sink_;
    AudioIo io_;
    std::unique_ptr<Session> session_;
    std::atomic<IEventListener*> target_{nullptr};
    std::atomic<bool> ready_{false};
    std::atomic<bool> finished_{false};
    std::string done_;   ///< written by run() before finished_
    std::string error_;  ///< likewise
    std::mutex stop_mu_;
    std::thread runner_;
};

}  // namespace server_detail

using server_detail::kSessionRate;
using server_detail::LiveSession;

struct SessionStream::Impl final : public IEventListener {
    Impl(OutputChannel& out, std::unique_ptr<LiveSession> live, int client_rate, int output_rate)
        : out_(out), session_(std::move(live)), client_rate_(client_rate), output_rate_(output_rate) {
        if (client_rate != kSessionRate) resampler_ = std::make_unique<Resampler>(client_rate, kSessionRate);
        session_->attach(this);
        pump_ = std::thread([this] { pump(); });
    }
    ~Impl() override { stop(); }

    void on_event(std::string_view event) override {
        const std::lock_guard<std::mutex> lock(outbox_mu_);
        outbox_.emplace_back(event);
    }

    void audio(std::string_view pcm) {
        if (ended_.load()) return;
        samples_.resize(pcm.size() / 2);
        for (std::size_t i = 0; i < samples_.size(); ++i) {
            const auto lo = static_cast<unsigned>(static_cast<std::uint8_t>(pcm[2 * i]));
            const auto hi = static_cast<unsigned>(static_cast<std::uint8_t>(pcm[2 * i + 1]));
            samples_[i] = static_cast<float>(static_cast<std::int16_t>(lo | (hi << 8))) / 32768.0f;
        }
        if (!resampler_) {
            session_->source().push(samples_);
            return;
        }
        resampled_.clear();
        resampler_->process(samples_, resampled_);
        session_->source().push(resampled_);
    }

    void end() {
        if (ended_.exchange(true)) return;
        if (resampler_) {
            resampled_.clear();
            resampler_->flush(resampled_);
            session_->source().push(resampled_);
        }
        session_->source().close();
    }

    void stop() {
        const std::lock_guard<std::mutex> lock(stop_mu_);
        stopping_.store(true);
        session_->stop();
        if (pump_.joinable()) pump_.join();
    }

    void wait() {
        std::unique_lock<std::mutex> lock(pumped_mu_);
        pumped_cv_.wait(lock, [this] { return pumped_; });
    }

private:
    void pump() {
        run_pump();
        {
            const std::lock_guard<std::mutex> lock(pumped_mu_);
            pumped_ = true;
        }
        pumped_cv_.notify_all();
    }

    void run_pump() {
        using std::chrono::milliseconds;
        const auto block = static_cast<std::size_t>(output_rate_ / 100);  // 10 ms
        std::vector<float> out(block);
        std::vector<std::int16_t> pcm;
        std::deque<std::string> batch;
        bool announced = false;
        auto next = std::chrono::steady_clock::now();
        for (;;) {
            // Read before the outbox: whatever the session posted before finishing goes out below.
            const bool finished = session_->finished();
            if (!announced && session_->ready()) {  // models loaded: a warm session is at once
                const PipelineSpec& spec = session_->spec();
                out_.ready({client_rate_, output_rate_, spec.source_language, spec.target_language});
                announced = true;
            }
            {
                const std::lock_guard<std::mutex> lock(outbox_mu_);
                batch.swap(outbox_);
            }
            for (const std::string& event : batch) out_.event(event);
            batch.clear();
            if (stopping_.load()) return;
            if (finished && session_->sink().queued() == 0) {
                if (session_->error().empty()) {
                    out_.done(session_->done());
                } else {
                    out_.error(session_->error());
                }
                return;
            }
            const std::size_t played = session_->sink().pull(out);
            if (played > 0) {
                pcm.resize(played);
                for (std::size_t i = 0; i < played; ++i) {
                    pcm[i] = static_cast<std::int16_t>(std::lround(std::clamp(out[i], -1.0f, 1.0f) * 32767.0f));
                }
                out_.audio(pcm);
            }
            next += milliseconds(10);
            const auto now = std::chrono::steady_clock::now();
            if (now - next > milliseconds(100)) next = now;  // fell behind (a slow send): resync
            std::this_thread::sleep_until(next);
        }
    }

    OutputChannel& out_;
    std::unique_ptr<LiveSession> session_;
    int client_rate_;
    int output_rate_;
    std::unique_ptr<Resampler> resampler_;
    std::vector<float> samples_;
    std::vector<float> resampled_;
    std::mutex outbox_mu_;
    std::deque<std::string> outbox_;
    std::atomic<bool> ended_{false};
    std::atomic<bool> stopping_{false};
    std::mutex stop_mu_;
    std::mutex pumped_mu_;
    std::condition_variable pumped_cv_;
    bool pumped_ = false;
    std::thread pump_;
};

SessionStream::SessionStream(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
SessionStream::~SessionStream() = default;
void SessionStream::audio(std::string_view pcm) { impl_->audio(pcm); }
void SessionStream::end() { impl_->end(); }
void SessionStream::wait() { impl_->wait(); }

struct SessionPool::Impl {
    explicit Impl(Options o) : options(std::move(o)) {}

    /// Keeps options.warm_sessions sessions loading or loaded. Called with mu held.
    void replenish() {
        while (warm.size() < options.warm_sessions) {
            try {
                warm.push_back(std::make_unique<LiveSession>(options));
            } catch (const std::exception& e) {
                log::error("server: cannot prepare a session: ", e.what());
                return;
            }
        }
    }

    /// The readiest warm session, else one still loading, else a new one. Called with mu held.
    std::unique_ptr<LiveSession> take() {
        // A warm session that failed to build (a missing model, say) is dropped: the new one
        // reports the error to the client.
        warm.erase(std::remove_if(warm.begin(), warm.end(),
                                  [](const std::unique_ptr<LiveSession>& s) { return s->finished(); }),
                   warm.end());
        auto it = std::find_if(warm.begin(), warm.end(), [](const std::unique_ptr<LiveSession>& s) { return s->ready(); });
        if (it == warm.end() && !warm.empty()) it = warm.begin();
        if (it == warm.end()) return std::make_unique<LiveSession>(options);  // cold: a bad config throws here
        std::unique_ptr<LiveSession> session = std::move(*it);
        warm.erase(it);
        return session;  // no replacement yet: that waits until a session ends
    }

    Options options;
    mutable std::mutex mu;
    std::deque<std::unique_ptr<LiveSession>> warm;      ///< ready, or loading
    std::vector<std::shared_ptr<SessionStream>> active;
    bool started = false;
};

SessionPool::SessionPool(Options options) : impl_(std::make_unique<Impl>(std::move(options))) {}

SessionPool::~SessionPool() { stop(); }

void SessionPool::start() {
    const std::lock_guard<std::mutex> lock(impl_->mu);
    impl_->started = true;
    impl_->replenish();
}

void SessionPool::stop() {
    std::vector<std::shared_ptr<SessionStream>> active;
    std::deque<std::unique_ptr<LiveSession>> warm;
    {
        const std::lock_guard<std::mutex> lock(impl_->mu);
        impl_->started = false;  // streams closing from here on are not replaced
        active.swap(impl_->active);
        warm.swap(impl_->warm);
    }
    for (const auto& s : active) s->impl_->stop();
    warm.clear();  // each stops and joins
}

std::shared_ptr<SessionStream> SessionPool::open(OutputChannel& out, int client_rate) {
    if (client_rate < 8000 || client_rate > 192000) {
        throw Refused(Refused::Reason::BadRate, "the sample rate must be 8000-192000 Hz");
    }
    const std::lock_guard<std::mutex> lock(impl_->mu);
    if (impl_->active.size() >= impl_->options.max_sessions) {
        throw Refused(Refused::Reason::Busy,
                      "busy: " + std::to_string(impl_->options.max_sessions) + " session(s) already running");
    }
    std::unique_ptr<LiveSession> live;
    try {
        live = impl_->take();
    } catch (const std::exception& e) {
        throw Refused(Refused::Reason::Failed, e.what());
    }
    std::shared_ptr<SessionStream> stream(new SessionStream(
        std::make_unique<SessionStream::Impl>(out, std::move(live), client_rate, impl_->options.output_rate)));
    impl_->active.push_back(stream);
    return stream;
}

void SessionPool::close(const std::shared_ptr<SessionStream>& stream) {
    if (!stream) return;
    bool found = false;
    {
        const std::lock_guard<std::mutex> lock(impl_->mu);
        const auto it = std::find(impl_->active.begin(), impl_->active.end(), stream);
        if (it != impl_->active.end()) {
            impl_->active.erase(it);
            found = true;
        }
    }
    stream->impl_->stop();
    const std::lock_guard<std::mutex> lock(impl_->mu);
    if (found && impl_->started) impl_->replenish();  // the next client finds its models loaded
}

std::size_t SessionPool::active() const {
    const std::lock_guard<std::mutex> lock(impl_->mu);
    return impl_->active.size();
}

std::size_t SessionPool::warm_ready() const {
    const std::lock_guard<std::mutex> lock(impl_->mu);
    return static_cast<std::size_t>(std::count_if(impl_->warm.begin(), impl_->warm.end(), [](const auto& s) {
        return s->ready() && !s->finished();
    }));
}

const SessionPool::Options& SessionPool::options() const noexcept { return impl_->options; }

}  // namespace ee
