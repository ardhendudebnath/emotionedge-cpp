#pragma once

#include <atomic>
#include <chrono>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "core/audio/audio_io.hpp"
#include "core/audio/echo_sim.hpp"
#include "core/pipeline/recorder.hpp"
#include "core/runtime/config.hpp"
#include "core/telemetry/telemetry.hpp"

namespace ee {

enum class RunMode {
    Offline,   ///< deterministic, as fast as possible: file in, file out
    Realtime,  ///< threaded; the input is paced at real time and playout drained in real time
    Live,      ///< threaded on caller-provided devices (AudioIo) until stop()
};

struct SessionOptions {
    std::filesystem::path config;
    std::vector<std::pair<std::string, std::string>> overrides;  ///< "stage.param" -> value
    RunMode mode = RunMode::Offline;

    // Offline / Realtime input: a WAV file or samples in memory.
    std::filesystem::path input_wav;
    std::vector<float> input_samples;
    int input_rate = 0;
    double speed = 1.0;  ///< Realtime pacing (1 = real time)
    /// Realtime: what is played comes back into the input through a simulated loudspeaker and
    /// room, as with an open speaker next to the microphone (exercises the echo canceller).
    std::optional<EchoSimulator::Config> echo_sim;

    AudioIo* live_io = nullptr;  ///< Live mode: capture/playback wired to devices by the caller
    /// Live mode: called once the graph is built (models loaded) and running.
    std::function<void()> on_ready;
    /// Any mode: each result as it arrives (RecorderStage's events). Must outlive run().
    IEventListener* events = nullptr;

    // Output (Offline / Realtime).
    int output_rate = 24000;
    std::filesystem::path output_wav;
    float source_mix_db = -120.0f;  ///< > -100: mix the original speech under the translation
    float duck_db = -12.0f;          ///< extra attenuation of the original while translation plays

    std::chrono::milliseconds timeout{0};  ///< Realtime/Live guard; 0 = derived from the input
};

struct SessionResult {
    bool completed = false;
    std::vector<UtteranceRecord> utterances;
    std::vector<float> output_audio;
    int output_rate = 0;
    double input_seconds = 0.0;
    std::uint64_t dropped_frames = 0;
};

/// Builds and runs one pipeline session: loads the YAML graph and the model manifest, wires the
/// audio endpoints for the run mode, runs the graph and collects captions, telemetry and the
/// translated audio (optionally mixed over the ducked original).
class Session {
public:
    explicit Session(SessionOptions options);
    ~Session();

    SessionResult run();
    /// Stops a Realtime/Live run from another thread (e.g. Ctrl+C).
    void stop() noexcept { stop_requested_.store(true); }

    [[nodiscard]] telemetry::Telemetry& telemetry() noexcept { return *telemetry_; }
    [[nodiscard]] const PipelineSpec& spec() const noexcept { return spec_; }

private:
    SessionOptions options_;
    PipelineSpec spec_;
    std::unique_ptr<telemetry::Telemetry> telemetry_;
    std::atomic<bool> stop_requested_{false};
};

/// Mixes `source` (resampled to `rate`) under `translation`, ducking it while translation plays.
[[nodiscard]] std::vector<float> mix_with_ducking(const std::vector<float>& translation, const std::vector<float>& source,
                                                  int source_rate, int rate, float source_db, float duck_db);

}  // namespace ee
