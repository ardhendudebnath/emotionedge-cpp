#include "core/runtime/log.hpp"

#include <atomic>
#include <cstdio>
#include <mutex>

namespace ee::log {

namespace {
std::atomic<Level> g_level{Level::Info};
std::mutex g_mutex;
std::function<void(Level, std::string_view)> g_sink;

std::string_view tag(Level level) {
    switch (level) {
    case Level::Debug: return "debug";
    case Level::Info: return "info";
    case Level::Warn: return "warn";
    case Level::Error: return "error";
    case Level::Off: return "off";
    }
    return "?";
}
}  // namespace

void set_level(Level lvl) noexcept { g_level.store(lvl, std::memory_order_relaxed); }

Level level() noexcept { return g_level.load(std::memory_order_relaxed); }

std::optional<Level> parse_level(std::string_view name) noexcept {
    for (Level lvl : {Level::Debug, Level::Info, Level::Warn, Level::Error, Level::Off}) {
        if (tag(lvl) == name) return lvl;
    }
    return std::nullopt;
}

void set_sink(std::function<void(Level, std::string_view)> sink) {
    std::lock_guard lock(g_mutex);
    g_sink = std::move(sink);
}

void write(Level lvl, std::string_view message) {
    std::lock_guard lock(g_mutex);
    if (g_sink) {
        g_sink(lvl, message);
        return;
    }
    const std::string_view t = tag(lvl);
    std::fprintf(stderr, "[%.*s] %.*s\n", static_cast<int>(t.size()), t.data(),
                 static_cast<int>(message.size()), message.data());
}

}  // namespace ee::log
