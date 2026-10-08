#pragma once

#include <memory>
#include <string>

#include "core/server/security.hpp"
#include "core/server/session_pool.hpp"

namespace ee {

/// Blueprint 5.3's WebSocket API. Each connection is one live session from the SessionPool,
/// which it shares with the gRPC server in the same process: the same warm sessions and the
/// same limit. The client streams speech in; it gets back each result as the pipeline
/// produces it, and the translated speech at playback pace.
///
/// Protocol, on ws://HOST:PORT/?rate=16000 (`rate`: the client's sample rate, default 16000):
///   client -> server
///     binary  mono 16-bit little-endian PCM at `rate`
///     text    {"type": "end"}: no more speech; finish, play out, then send "done"
///   server -> client
///     text    {"type": "ready", "input_rate", "output_rate", "source_language", "target_language"}
///             once the models are loaded (at once from a warm session; audio sent before it is
///             buffered, up to 30 s);
///             then the recorder's events (core/pipeline/recorder.hpp, event_json): transcript,
///             emotion, translation, prosody, consistency and playout;
///             {"type": "done", "session": {...}} with the session export, after the last audio;
///             {"type": "error", "message"} when the session cannot start or fails
///     binary  the translated speech, mono 16-bit little-endian PCM at output_rate, sent in 10 ms
///             blocks as it would play. Silence is not sent.
///
/// With ServerSecurity: wss:// with the certificate (needs a build with OpenSSL), and a token the
/// client presents as "Authorization: Bearer <token>" or, from browsers, which cannot set that
/// header, as ?token=. Without it the connection is closed (1008) after an "unauthorized" error.
class WebSocketServer {
public:
    WebSocketServer(SessionPool& pool, std::string host, int port, ServerSecurity security = {});
    ~WebSocketServer();
    WebSocketServer(const WebSocketServer&) = delete;
    WebSocketServer& operator=(const WebSocketServer&) = delete;

    /// Binds and starts accepting connections. Throws std::runtime_error if it cannot listen, and
    /// ConfigError for unusable TLS settings.
    void start();
    /// Ends its clients' sessions, then stops listening.
    void stop();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ee
