#include "core/server/websocket_server.hpp"

#include <ixwebsocket/IXNetSystem.h>
#include <ixwebsocket/IXWebSocketServer.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <map>
#include <mutex>
#include <stdexcept>

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

std::string error_json(std::string_view message) {
    return json{{"type", "error"}, {"message", std::string(message)}}.dump(-1, ' ', false, json::error_handler_t::replace);
}

/// A session's output as WebSocket frames: JSON text for the results, binary for the speech.
class WebSocketChannel final : public OutputChannel {
public:
    explicit WebSocketChannel(ix::WebSocket& ws) : ws_(ws) {}

    void ready(const Ready& r) override {
        ws_.sendText(json{{"type", "ready"},
                          {"input_rate", r.input_rate},
                          {"output_rate", r.output_rate},
                          {"source_language", r.source_language},
                          {"target_language", r.target_language}}
                         .dump());
    }
    void event(std::string_view json_text) override { ws_.sendText(std::string(json_text)); }
    void audio(std::span<const std::int16_t> pcm) override {
        std::string bytes(pcm.size() * 2, '\0');
        for (std::size_t i = 0; i < pcm.size(); ++i) {
            const auto s = static_cast<std::uint16_t>(pcm[i]);
            bytes[2 * i] = static_cast<char>(s & 0xFFu);
            bytes[2 * i + 1] = static_cast<char>(s >> 8);
        }
        ws_.sendBinary(bytes);
    }
    void done(std::string_view session_json) override {
        ws_.sendText(R"({"type":"done","session":)" + std::string(session_json) + "}");
    }
    void error(std::string_view message) override { ws_.sendText(error_json(message)); }

private:
    ix::WebSocket& ws_;
};

/// One connection: its channel lives as long as its stream does.
struct Client {
    explicit Client(ix::WebSocket& ws) : channel(ws) {}
    WebSocketChannel channel;
    std::shared_ptr<SessionStream> stream;
};

}  // namespace

struct WebSocketServer::Impl {
    Impl(SessionPool& p, std::string h, int pt) : pool(p), host(std::move(h)), port(pt), server(pt, host) {}

    void on_message(const std::shared_ptr<ix::ConnectionState>& state, ix::WebSocket& ws,
                    const ix::WebSocketMessagePtr& msg) {
        switch (msg->type) {
        case ix::WebSocketMessageType::Open: open(state->getId(), ws, msg->openInfo.uri); break;
        case ix::WebSocketMessageType::Message: {
            const std::shared_ptr<Client> c = find(state->getId());
            if (c == nullptr) return;
            if (msg->binary) {
                c->stream->audio(msg->str);
                break;
            }
            const json control = json::parse(msg->str, nullptr, false);
            if (control.is_object() && control.value("type", "") == "end") {
                c->stream->end();
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
        auto client = std::make_shared<Client>(ws);
        try {
            client->stream = pool.open(client->channel, rate);
        } catch (const SessionPool::Refused& refused) {
            const std::string reason = refused.what();
            ws.sendText(error_json(reason));
            ws.close(1013, reason.substr(0, 120));  // 1013: try again later; reasons are <= 123 bytes
            return;
        }
        {
            const std::lock_guard<std::mutex> lock(mu);
            clients[id] = client;
        }
        log::info("websocket: session ", id, " opened (", rate, " Hz in)");
    }

    std::shared_ptr<Client> find(const std::string& id) {
        const std::lock_guard<std::mutex> lock(mu);
        const auto it = clients.find(id);
        return it == clients.end() ? nullptr : it->second;
    }

    void close(const std::string& id) {
        std::shared_ptr<Client> c;
        {
            const std::lock_guard<std::mutex> lock(mu);
            const auto it = clients.find(id);
            if (it == clients.end()) return;
            c = std::move(it->second);
            clients.erase(it);
        }
        pool.close(c->stream);
        log::info("websocket: session ", id, " closed");
    }

    SessionPool& pool;
    std::string host;
    int port;
    ix::WebSocketServer server;
    std::mutex mu;
    std::map<std::string, std::shared_ptr<Client>> clients;  ///< by ix::ConnectionState id
    bool started = false;
};

WebSocketServer::WebSocketServer(SessionPool& pool, std::string host, int port) {
    ix::initNetSystem();
    impl_ = std::make_unique<Impl>(pool, std::move(host), port);
}

WebSocketServer::~WebSocketServer() { stop(); }

void WebSocketServer::start() {
    Impl& g = *impl_;
    g.server.disablePerMessageDeflate();
    g.server.setOnClientMessageCallback(
        [impl = impl_.get()](std::shared_ptr<ix::ConnectionState> state, ix::WebSocket& ws, const ix::WebSocketMessagePtr& msg) {
            impl->on_message(state, ws, msg);
        });
    const auto [ok, error] = g.server.listen();
    if (!ok) throw std::runtime_error("cannot listen on " + g.host + ":" + std::to_string(g.port) + ": " + error);
    g.server.start();
    g.started = true;
}

void WebSocketServer::stop() {
    if (!impl_ || !impl_->started) return;
    std::map<std::string, std::shared_ptr<Client>> all;
    {
        const std::lock_guard<std::mutex> lock(impl_->mu);
        all.swap(impl_->clients);
    }
    for (auto& [id, c] : all) impl_->pool.close(c->stream);  // joins the stream before its channel goes
    impl_->server.stop();
    impl_->started = false;
}

}  // namespace ee
