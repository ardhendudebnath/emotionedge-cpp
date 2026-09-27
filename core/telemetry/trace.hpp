#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <string_view>

namespace ee::telemetry {

/// One complete ("ph":"X") trace event: a stage handling one frame.
struct TraceEvent {
    const char* name = "";      ///< stage name (must outlive the writer)
    const char* category = "";  ///< frame kind
    std::int64_t ts_us = 0;
    std::int64_t dur_us = 0;
    std::uint32_t tid = 0;
    std::uint64_t utterance = 0;
};

/// Streams events as Chrome trace-event JSON, which the Perfetto UI (ui.perfetto.dev) opens
/// directly. Single writer: the telemetry thread (T7), or the caller in deterministic runs.
class TraceWriter {
public:
    explicit TraceWriter(const std::string& path);
    ~TraceWriter();
    TraceWriter(const TraceWriter&) = delete;
    TraceWriter& operator=(const TraceWriter&) = delete;

    [[nodiscard]] bool ok() const { return static_cast<bool>(out_); }
    void write(const TraceEvent& event);
    void thread_name(std::uint32_t tid, std::string_view name);
    void close();

private:
    void separator();
    std::ofstream out_;
    bool first_ = true;
    bool closed_ = false;
};

}  // namespace ee::telemetry
