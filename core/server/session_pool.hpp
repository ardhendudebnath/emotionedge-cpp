#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ee {

/// Where a live session's results go: one implementation per transport (WebSocket, gRPC).
/// Called from the stream's own thread only, in order: ready() first, done() or error() last.
class OutputChannel {
public:
    struct Ready {
        int input_rate = 0;
        int output_rate = 0;
        std::string source_language;
        std::string target_language;
    };

    virtual ~OutputChannel() = default;
    /// The session's models are loaded (at once for a warm session).
    virtual void ready(const Ready& ready) = 0;
    /// One of the recorder's live events (event_json in core/pipeline/recorder.hpp).
    virtual void event(std::string_view json) = 0;
    /// The translated speech: mono 16-bit PCM at Ready::output_rate, 10 ms at a time, at
    /// playback pace. Silence is not sent.
    virtual void audio(std::span<const std::int16_t> pcm) = 0;
    /// The session's report (to_session_json), after the last audio.
    virtual void done(std::string_view session_json) = 0;
    virtual void error(std::string_view message) = 0;
};

/// One client on one live session: its speech in, its OutputChannel out. The session's playback
/// is drained at real-time pace, as a sound card would take it, so the pipeline's playout timing
/// (adaptive pacing, barge-in) works as with a device. Made by SessionPool::open.
class SessionStream {
public:
    ~SessionStream();
    SessionStream(const SessionStream&) = delete;
    SessionStream& operator=(const SessionStream&) = delete;

    /// The client's speech: mono 16-bit little-endian PCM bytes at its rate. From one thread.
    void audio(std::string_view pcm);
    /// No more speech: the session finishes what it heard and plays it out, then done().
    void end();
    /// Blocks until done() or error() has been sent, or the stream was stopped.
    void wait();

    struct Impl;

private:
    friend class SessionPool;
    explicit SessionStream(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

/// The live pipeline sessions behind a server (5.3), shared by its transports.
/// - At most max_sessions clients at once.
/// - Loading a session's models takes seconds, so warm_sessions are kept loaded. A client takes
///   one and is ready at once. A replacement loads when a session ends: loading one during a
///   session slowed it (300-450 ms on the p95 of half the runs, on jfk.wav).
/// - A session serves one client only: it learns that speaker (voice print, emotion state,
///   closed-loop correction).
/// - Sessions take speech at 16 kHz; streams resample other rates on the way in.
class SessionPool {
public:
    struct Options {
        std::filesystem::path config;
        std::vector<std::pair<std::string, std::string>> overrides;  ///< "stage.param" -> value
        /// Each session loads its own models: on the GPU about 3 GB of memory each.
        std::size_t max_sessions = 1;
        /// Sessions kept loaded for the next clients, on top of max_sessions in memory.
        /// 0 = load on connect (seconds before ready).
        std::size_t warm_sessions = 1;
        int output_rate = 24000;
    };

    /// Why open() refused a client. Transports map it to their own status codes.
    class Refused : public std::runtime_error {
    public:
        enum class Reason { Busy, BadRate, Failed };
        Refused(Reason reason, const std::string& message) : std::runtime_error(message), reason_(reason) {}
        [[nodiscard]] Reason reason() const noexcept { return reason_; }

    private:
        Reason reason_;
    };

    explicit SessionPool(Options options);
    ~SessionPool();
    SessionPool(const SessionPool&) = delete;
    SessionPool& operator=(const SessionPool&) = delete;

    /// Starts loading the warm sessions.
    void start();
    /// Ends every stream and warm session. Streams still held by transports stay valid (stopped).
    void stop();

    /// A stream for a new client whose speech comes at `client_rate` Hz. `out` must outlive it.
    /// Throws Refused.
    [[nodiscard]] std::shared_ptr<SessionStream> open(OutputChannel& out, int client_rate);
    /// The client is gone, finished or not: ends its stream, then refills the warm sessions.
    void close(const std::shared_ptr<SessionStream>& stream);

    [[nodiscard]] std::size_t active() const;
    /// Warm sessions whose models are loaded, waiting for a client.
    [[nodiscard]] std::size_t warm_ready() const;
    [[nodiscard]] const Options& options() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ee
