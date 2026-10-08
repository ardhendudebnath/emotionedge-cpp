#include "core/server/websocket_server.hpp"

#include <ixwebsocket/IXNetSystem.h>
#include <ixwebsocket/IXWebSocketServer.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <map>
#include <mutex>
#include <stdexcept>

#include "core/runtime/log.hpp"
#include "core/runtime/params.hpp"  // ConfigError

namespace ee {

namespace {

using json = nlohmann::json;

/// `key`'s value in a request URI's query ("/?rate=48000&token=..."), percent-decoded; empty if
/// absent.
std::string query_value(const std::string& uri, const std::string& key) {
    const auto q = uri.find('?');
    if (q == std::string::npos) return {};
    std::size_t pos = q + 1;
    while (pos < uri.size()) {
        const std::size_t amp = std::min(uri.find('&', pos), uri.size());
        const std::string pair = uri.substr(pos, amp - pos);
        const auto eq = pair.find('=');
        if (eq != std::string::npos && pair.substr(0, eq) == key) {
            std::string value;
            const std::string raw = pair.substr(eq + 1);
            for (std::size_t i = 0; i < raw.size(); ++i) {
                if (raw[i] == '%' && i + 2 < raw.size() && std::isxdigit(static_cast<unsigned char>(raw[i + 1])) &&
                    std::isxdigit(static_cast<unsigned char>(raw[i + 2]))) {
                    value.push_back(static_cast<char>(std::stoi(raw.substr(i + 1, 2), nullptr, 16)));
                    i += 2;
                } else {
                    value.push_back(raw[i] == '+' ? ' ' : raw[i]);
                }
            }
            return value;
        }
        pos = amp + 1;
    }
    return {};
}

int query_int(const std::string& uri, const std::string& key, int fallback) {
    const std::string value = query_value(uri, key);
    if (value.empty()) return fallback;
    try {
        return std::stoi(value);
    } catch (const std::exception&) {
        return fallback;
    }
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
    Impl(SessionPool& p, std::string h, int pt, ServerSecurity s)
        : pool(p), host(std::move(h)), port(pt), security(std::move(s)), server(pt, host) {}

    void on_message(const std::shared_ptr<ix::ConnectionState>& state, ix::WebSocket& ws,
                    const ix::WebSocketMessagePtr& msg) {
        switch (msg->type) {
        case ix::WebSocketMessageType::Open: open(state->getId(), ws, msg->openInfo); break;
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

    void open(const std::string& id, ix::WebSocket& ws, const ix::WebSocketOpenInfo& info) {
        if (!security.token.empty()) {
            // The header from programs; ?token= from browsers, which cannot set it.
            std::string presented;
            if (const auto it = info.headers.find("Authorization"); it != info.headers.end()) {
                presented = std::string(bearer_token(it->second));
            }
            if (presented.empty()) presented = query_value(info.uri, "token");
            if (!token_matches(security.token, presented)) {
                ws.sendText(error_json("unauthorized: this server needs a valid token"));
                ws.close(1008, "unauthorized");  // 1008: policy violation
                log::warn("websocket: refused connection ", id, ": no valid token");
                return;
            }
        }
        const int rate = query_int(info.uri, "rate", 16000);
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
    ServerSecurity security;
    ix::WebSocketServer server;
    std::mutex mu;
    std::map<std::string, std::shared_ptr<Client>> clients;  ///< by ix::ConnectionState id
    bool started = false;
};

WebSocketServer::WebSocketServer(SessionPool& pool, std::string host, int port, ServerSecurity security) {
    ix::initNetSystem();
    impl_ = std::make_unique<Impl>(pool, std::move(host), port, std::move(security));
}

WebSocketServer::~WebSocketServer() { stop(); }

void WebSocketServer::start() {
    Impl& g = *impl_;
    g.security.validate();
    if (g.security.tls()) {
#if defined(IXWEBSOCKET_USE_TLS)
        ix::SocketTLSOptions tls;
        tls.tls = true;
        tls.certFile = g.security.cert_file.string();
        tls.keyFile = g.security.key_file.string();
        tls.caFile = "NONE";  // clients are not asked for certificates
        g.server.setTLSOptions(tls);
#else
        throw ConfigError("this build's WebSocket server has no TLS (OpenSSL was not found when it was configured)");
#endif
    }
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
