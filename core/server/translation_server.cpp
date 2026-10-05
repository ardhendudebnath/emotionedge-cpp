#include "core/server/translation_server.hpp"

#include <ixwebsocket/IXNetSystem.h>
#include <ixwebsocket/IXWebSocketServer.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>

#include "core/audio/audio_io.hpp"
#include "core/pipeline/recorder.hpp"
#include "core/pipeline/session.hpp"
#include "core/runtime/log.hpp"

namespace ee {

namespace {

using json = nlohmann::json;

/// `key`'s integer value in a request URI's query ("/?rate=48000"), or `fallback`.
int query_int(const std::string& uri, const std::string& key, int fallback) {
    const auto q = uri.find('?');
    if (q == std::string::npos) return fallback;
    std::size_t pos = q + 1;
    while (pos < uri.size()) {
        const std::size_t amp = std::min(uri.find('&', pos), uri.size());
        const std::string pair = uri.substr(pos, amp - pos);
        const auto eq = pair.find('=');
        if (eq != std::string::npos && pair.substr(0, eq) == key) {
            try {
                return std::stoi(pair.substr(eq + 1));
            } catch (const std::exception&) {
                return fallback;
            }
        }
        pos = amp + 1;
    }
    return fallback;
}

std::string error_json(const std::string& message) {
    return json{{"type", "error"}, {"message", message}}.dump(-1, ' ', false, json::error_handler_t::replace);
}

/// One client: a live Session fed from the socket. Its playback is drained back to the client at
/// real-time pace, as a speaker would, so the pipeline's playout timing (adaptive pacing,
/// barge-in) works as with a device. Events and audio go out from one thread, in order.
class Connection final : public IEventListener {
public:
    Connection(ix::WebSocket& ws, const TranslationServer::Options& options, int input_rate)
        : ws_(ws),
          output_rate_(options.output_rate),
          source_(input_rate, static_cast<std::size_t>(input_rate) * 30),
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
        so.on_ready = [this, input_rate] {
            const PipelineSpec& spec = session_->spec();
            post(json{{"type", "ready"},
                      {"input_rate", input_rate},
                      {"output_rate", output_rate_},
                      {"source_language", spec.source_language},
                      {"target_language", spec.target_language}}
                     .dump());
        };
        session_ = std::make_unique<Session>(std::move(so));  // a bad config throws here
        runner_ = std::thread([this] { run(); });
        pump_ = std::thread([this] { pump(); });
    }
    ~Connection() override { stop(); }
    Connection(const Connection&) = delete;
    Connection& operator=(const Connection&) = delete;

    void on_event(std::string_view event) override { post(std::string(event)); }

    /// The client's speech: 16-bit little-endian PCM. Called from the socket's thread only.
    void audio(const std::string& pcm) {
        if (ended_.load()) return;
        samples_.resize(pcm.size() / 2);
        for (std::size_t i = 0; i < samples_.size(); ++i) {
            const auto lo = static_cast<unsigned>(static_cast<std::uint8_t>(pcm[2 * i]));
            const auto hi = static_cast<unsigned>(static_cast<std::uint8_t>(pcm[2 * i + 1]));
            samples_[i] = static_cast<float>(static_cast<std::int16_t>(lo | (hi << 8))) / 32768.0f;
        }
        source_.push(samples_);
    }

    /// No more speech: the session finishes what it heard and plays it out, then reports.
    void end() noexcept {
        ended_.store(true);
        source_.close();
    }

    /// Ends the session now and joins its threads. Idempotent.
    void stop() {
        const std::lock_guard<std::mutex> lock(stop_mu_);
        stopping_.store(true);
        end();
        session_->stop();
        if (runner_.joinable()) runner_.join();
        if (pump_.joinable()) pump_.join();
    }

private:
    void post(std::string message) {
        const std::lock_guard<std::mutex> lock(outbox_mu_);
        outbox_.push_back(std::move(message));
    }

    void run() {
        try {
            const SessionResult result = session_->run();
            done_ = R"({"type":"done","session":)" +
                    to_session_json(result.utterances, session_->spec(), &session_->telemetry()) + "}";
        } catch (const std::exception& e) {
            log::error("server: session failed: ", e.what());
            post(error_json(e.what()));
        }
        finished_.store(true, std::memory_order_release);
    }

    void pump() {
        using std::chrono::milliseconds;
        const auto block = static_cast<std::size_t>(output_rate_ / 100);  // 10 ms
        std::vector<float> out(block);
        std::string pcm;
        std::deque<std::string> batch;
        auto next = std::chrono::steady_clock::now();
        for (;;) {
            // Read before the outbox: whatever the session posted before finishing goes out below.
            const bool finished = finished_.load(std::memory_order_acquire);
            {
                const std::lock_guard<std::mutex> lock(outbox_mu_);
                batch.swap(outbox_);
            }
            for (const std::string& message : batch) ws_.sendText(message);
            batch.clear();
            if (stopping_.load()) return;
            if (finished && sink_.queued() == 0) {
                if (!done_.empty()) ws_.sendText(done_);
                return;
            }
            const std::size_t played = sink_.pull(out);
            if (played > 0) {
                pcm.resize(played * 2);
                for (std::size_t i = 0; i < played; ++i) {
                    const auto s = static_cast<std::int16_t>(std::lround(std::clamp(out[i], -1.0f, 1.0f) * 32767.0f));
                    pcm[2 * i] = static_cast<char>(static_cast<std::uint16_t>(s) & 0xFFu);
                    pcm[2 * i + 1] = static_cast<char>(static_cast<std::uint16_t>(s) >> 8);
                }
                ws_.sendBinary(pcm);
            }
            next += milliseconds(10);
            const auto now = std::chrono::steady_clock::now();
            if (now - next > milliseconds(100)) next = now;  // fell behind (a slow send): resync
            std::this_thread::sleep_until(next);
        }
    }

    ix::WebSocket& ws_;
    int output_rate_;
    RingSource source_;
    RingSink sink_;
    AudioIo io_;
    std::unique_ptr<Session> session_;
    std::vector<float> samples_;
    std::mutex outbox_mu_;
    std::deque<std::string> outbox_;
    std::string done_;  ///< written by run() before finished_
    std::atomic<bool> finished_{false};
    std::atomic<bool> ended_{false};
    std::atomic<bool> stopping_{false};
    std::mutex stop_mu_;
    std::thread runner_;
    std::thread pump_;
};

}  // namespace

struct TranslationServer::Impl {
    explicit Impl(Options o) : options(std::move(o)), server(options.port, options.host) {}

    void on_message(const std::shared_ptr<ix::ConnectionState>& state, ix::WebSocket& ws,
                    const ix::WebSocketMessagePtr& msg) {
        switch (msg->type) {
        case ix::WebSocketMessageType::Open: open(state->getId(), ws, msg->openInfo.uri); break;
        case ix::WebSocketMessageType::Message: {
            const std::shared_ptr<Connection> c = find(state->getId());
            if (c == nullptr) return;
            if (msg->binary) {
                c->audio(msg->str);
                break;
            }
            const json control = json::parse(msg->str, nullptr, false);
            if (control.is_object() && control.value("type", "") == "end") {
                c->end();
            } else {
                ws.sendText(error_json("unknown message; send audio as binary PCM, or {\"type\": \"end\"}"));
            }
            break;
        }
        case ix::WebSocketMessageType::Close:
        case ix::WebSocketMessageType::Error: close(state->getId()); break;
        default: break;
        }
    }

    void open(const std::string& id, ix::WebSocket& ws, const std::string& uri) {
        const int rate = query_int(uri, "rate", 16000);
        std::string refused;
        {
            const std::lock_guard<std::mutex> lock(mu);
            if (rate < 8000 || rate > 192000) {
                refused = "rate must be 8000-192000 Hz";
            } else if (connections.size() >= options.max_sessions) {
                refused = "busy: " + std::to_string(options.max_sessions) + " session(s) already running";
            } else {
                try {
                    connections[id] = std::make_shared<Connection>(ws, options, rate);
                    log::info("server: session ", id, " opened (", rate, " Hz in)");
                } catch (const std::exception& e) {
                    refused = e.what();
                }
            }
        }
        if (!refused.empty()) {
            ws.sendText(error_json(refused));
            ws.close(1013, refused.substr(0, 120));  // 1013: try again later; reasons are <= 123 bytes
        }
    }

    std::shared_ptr<Connection> find(const std::string& id) {
        const std::lock_guard<std::mutex> lock(mu);
        const auto it = connections.find(id);
        return it == connections.end() ? nullptr : it->second;
    }

    void close(const std::string& id) {
        std::shared_ptr<Connection> c;
        {
            const std::lock_guard<std::mutex> lock(mu);
            const auto it = connections.find(id);
            if (it == connections.end()) return;
            c = std::move(it->second);
            connections.erase(it);
        }
        c->stop();
        log::info("server: session ", id, " closed");
    }

    Options options;
    ix::WebSocketServer server;
    mutable std::mutex mu;
    std::map<std::string, std::shared_ptr<Connection>> connections;  ///< by ix::ConnectionState id
    bool started = false;
};

TranslationServer::TranslationServer(Options options) {
    ix::initNetSystem();
    impl_ = std::make_unique<Impl>(std::move(options));
}

TranslationServer::~TranslationServer() { stop(); }

void TranslationServer::start() {
    Impl& g = *impl_;
    g.server.disablePerMessageDeflate();
    g.server.setOnClientMessageCallback(
        [impl = impl_.get()](std::shared_ptr<ix::ConnectionState> state, ix::WebSocket& ws, const ix::WebSocketMessagePtr& msg) {
            impl->on_message(state, ws, msg);
        });
    const auto [ok, error] = g.server.listen();
    if (!ok) throw std::runtime_error("cannot listen on " + g.options.host + ":" + std::to_string(g.options.port) + ": " + error);
    g.server.start();
    g.started = true;
}

void TranslationServer::stop() {
    if (!impl_ || !impl_->started) return;
    std::map<std::string, std::shared_ptr<Connection>> all;
    {
        const std::lock_guard<std::mutex> lock(impl_->mu);
        all.swap(impl_->connections);
    }
    for (auto& [id, c] : all) c->stop();
    impl_->server.stop();
    impl_->started = false;
}

std::size_t TranslationServer::active_sessions() const {
    const std::lock_guard<std::mutex> lock(impl_->mu);
    return impl_->connections.size();
}

}  // namespace ee
