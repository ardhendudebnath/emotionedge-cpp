#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace ee {

/// Blueprint 5.3 "gRPC / WebSocket": a WebSocket server that runs one live pipeline session per
/// connection. The client streams speech in; it gets back each result as the pipeline produces
/// it, and the translated speech at playback pace.
///
/// Protocol, on ws://HOST:PORT/?rate=16000 (`rate`: the client's sample rate, default 16000):
///   client -> server
///     binary  mono 16-bit little-endian PCM at `rate`
///     text    {"type": "end"}: no more speech; finish, play out, then send "done"
///   server -> client
///     text    {"type": "ready", "input_rate", "output_rate", "source_language", "target_language"}
///             once the models are loaded (audio sent before it is buffered, up to 30 s);
///             then the recorder's events (core/pipeline/recorder.hpp, event_json): transcript,
///             emotion, translation, prosody, consistency and playout;
///             {"type": "done", "session": {...}} with the session export, after the last audio;
///             {"type": "error", "message"} when the session cannot start or fails
///     binary  the translated speech, mono 16-bit little-endian PCM at output_rate, sent in 10 ms
///             blocks as it would play. Silence is not sent.
class TranslationServer {
public:
    struct Options {
        std::string host = "127.0.0.1";
        int port = 8080;
        /// Each session loads its own models: on the GPU about 3 GB of memory each.
        std::size_t max_sessions = 1;
        std::filesystem::path config;
        std::vector<std::pair<std::string, std::string>> overrides;  ///< "stage.param" -> value
        int output_rate = 24000;
    };

    explicit TranslationServer(Options options);
    ~TranslationServer();
    TranslationServer(const TranslationServer&) = delete;
    TranslationServer& operator=(const TranslationServer&) = delete;

    /// Binds and starts accepting connections. Throws std::runtime_error if it cannot listen.
    void start();
    /// Ends every session, then stops listening.
    void stop();
    [[nodiscard]] std::size_t active_sessions() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ee
