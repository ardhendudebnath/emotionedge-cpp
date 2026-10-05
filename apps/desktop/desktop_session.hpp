#pragma once

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "core/pipeline/live_view.hpp"

namespace ee::desktop {

/// Runs one live pipeline session for the window. The input is the microphone, or a WAV file
/// played in at speaking pace. The translation goes out of a playback device. Results go to a
/// LiveView, and the latency panel reads the session's telemetry. A session runs on its own
/// thread; the window only starts, stops and reads.
class DesktopSession {
public:
    struct Settings {
        std::filesystem::path config;
        std::vector<std::pair<std::string, std::string>> overrides;  ///< "stage.param" -> value
        std::string capture_device;   ///< substring of a device name; empty = the default
        std::string playback_device;
        std::filesystem::path input_wav;  ///< empty = the microphone
        bool mute = false;  ///< no playback device: the translation is paced but not played
    };
    struct LatencyRow {
        std::string name;
        double p50_ms = 0.0;
        double p95_ms = 0.0;
        double budget_ms = 0.0;
        std::uint64_t count = 0;
    };
    struct Stats {
        std::vector<LatencyRow> rows;
        LatencyRow end_to_end;
        std::uint64_t dropout_samples = 0;
        double playout_queued_s = 0.0;
    };

    explicit DesktopSession(LiveView& view);
    ~DesktopSession();
    DesktopSession(const DesktopSession&) = delete;
    DesktopSession& operator=(const DesktopSession&) = delete;

    /// Ends any running session, then starts a new one.
    void start(Settings settings);
    /// Asks the running session to end; returns at once.
    void stop() noexcept;
    [[nodiscard]] bool running() const noexcept { return running_.load(); }
    [[nodiscard]] std::string status() const;
    /// The current session's figures, or the last one's once it has ended.
    [[nodiscard]] Stats stats() const;

private:
    struct Run;
    void work(Run* run, Settings settings);
    void set_status(std::string status);

    LiveView& view_;
    mutable std::mutex mu_;
    std::unique_ptr<Run> run_;
    std::thread worker_;
    std::atomic<bool> running_{false};
    std::string status_ = "Idle.";
};

}  // namespace ee::desktop
