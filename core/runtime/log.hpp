#pragma once

#include <functional>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>

/// Minimal thread-safe logging. Not for the audio thread: stages log on setup and on rare events.
namespace ee::log {

enum class Level { Debug = 0, Info = 1, Warn = 2, Error = 3, Off = 4 };

void set_level(Level level) noexcept;
[[nodiscard]] Level level() noexcept;
[[nodiscard]] std::optional<Level> parse_level(std::string_view name) noexcept;

/// Replaces the default stderr sink (pass an empty function to restore it).
void set_sink(std::function<void(Level, std::string_view)> sink);

void write(Level level, std::string_view message);

template <typename... Args>
[[nodiscard]] std::string concat(const Args&... args) {
    std::ostringstream os;
    (os << ... << args);
    return os.str();
}

template <typename... Args>
void debug(const Args&... args) {
    if (level() <= Level::Debug) write(Level::Debug, concat(args...));
}
template <typename... Args>
void info(const Args&... args) {
    if (level() <= Level::Info) write(Level::Info, concat(args...));
}
template <typename... Args>
void warn(const Args&... args) {
    if (level() <= Level::Warn) write(Level::Warn, concat(args...));
}
template <typename... Args>
void error(const Args&... args) {
    if (level() <= Level::Error) write(Level::Error, concat(args...));
}

}  // namespace ee::log
