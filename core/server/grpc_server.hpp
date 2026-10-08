#pragma once

#include <memory>
#include <string>

#include "core/server/session_pool.hpp"

namespace ee {

/// Blueprint 5.3's gRPC API: the emotionedge.v1.Translator service
/// (proto/emotionedge/v1/translator.proto). Each Translate call is one live session from the
/// SessionPool, which it shares with the WebSocket server in the same process: the same warm
/// sessions and the same limit.
class GrpcServer {
public:
    /// port 0 picks a free port (see port()).
    GrpcServer(SessionPool& pool, std::string host, int port);
    ~GrpcServer();
    GrpcServer(const GrpcServer&) = delete;
    GrpcServer& operator=(const GrpcServer&) = delete;

    /// Binds and starts serving. Throws std::runtime_error if it cannot listen.
    void start();
    /// Stops serving; end the pool's sessions first, or in-flight calls finish first.
    void stop();
    /// The port it listens on, once started.
    [[nodiscard]] int port() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ee
