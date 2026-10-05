#include "apps/desktop/desktop_session.hpp"

#include <chrono>

#include "core/audio/audio_io.hpp"
#include "core/audio/device.hpp"
#include "core/audio/wav.hpp"
#include "core/pipeline/session.hpp"
#include "core/telemetry/telemetry.hpp"

namespace ee::desktop {

namespace {
constexpr int kOutputRate = 24000;
}

/// One session's audio endpoints and pipeline. The session is declared last, so it is destroyed
/// before the endpoints it uses.
struct DesktopSession::Run {
    std::unique_ptr<RingSource> source;
    std::unique_ptr<RingSink> sink;
    AudioIo io;
    std::unique_ptr<Session> session;
    std::atomic<bool> stop{false};
};

DesktopSession::DesktopSession(LiveView& view) : view_(view) {}

DesktopSession::~DesktopSession() {
    stop();
    if (worker_.joinable()) worker_.join();
}

void DesktopSession::start(Settings settings) {
    stop();
    if (worker_.joinable()) worker_.join();
    view_.clear();
    {
        const std::lock_guard<std::mutex> lock(mu_);
        run_ = std::make_unique<Run>();
    }
    running_.store(true);
    worker_ = std::thread([this, run = run_.get(), s = std::move(settings)]() mutable { work(run, std::move(s)); });
}

void DesktopSession::stop() noexcept {
    const std::lock_guard<std::mutex> lock(mu_);
    if (!run_) return;
    run_->stop.store(true);
    if (run_->session) run_->session->stop();
}

std::string DesktopSession::status() const {
    const std::lock_guard<std::mutex> lock(mu_);
    return status_;
}

void DesktopSession::set_status(std::string status) {
    const std::lock_guard<std::mutex> lock(mu_);
    status_ = std::move(status);
}

DesktopSession::Stats DesktopSession::stats() const {
    Stats out;
    const std::lock_guard<std::mutex> lock(mu_);
    if (!run_ || !run_->session) return out;
    const telemetry::Telemetry& t = run_->session->telemetry();
    for (const auto& row : t.budget_report()) {
        out.rows.push_back({std::string(row.row->name), static_cast<double>(row.snapshot.percentile(0.5)) / 1000.0,
                            static_cast<double>(row.snapshot.percentile(0.95)) / 1000.0,
                            static_cast<double>(row.row->target_us) / 1000.0, row.snapshot.count});
    }
    const auto e2e = t.end_to_end();
    out.end_to_end = {"End to end", static_cast<double>(e2e.percentile(0.5)) / 1000.0,
                      static_cast<double>(e2e.percentile(0.95)) / 1000.0, 800.0, e2e.count};
    out.dropout_samples = t.dropouts();
    out.playout_queued_s = run_->io.playout_queued_s.load(std::memory_order_relaxed);
    return out;
}

void DesktopSession::work(Run* run, Settings s) {
    try {
        const bool from_file = !s.input_wav.empty();
        std::vector<float> file;
        int input_rate = 16000;
        if (from_file) {
            WavData wav = read_wav(s.input_wav);
            file = std::move(wav.samples);
            input_rate = wav.sample_rate;
        }
        run->source = std::make_unique<RingSource>(input_rate, static_cast<std::size_t>(input_rate) * 30);
        run->sink = std::make_unique<RingSink>(kOutputRate, static_cast<std::size_t>(kOutputRate) * 60);
        run->io.capture = run->source.get();
        run->io.playback = run->sink.get();

        std::unique_ptr<PacedFeeder> feeder;
        if (from_file) {
            feeder = std::make_unique<PacedFeeder>(*run->source, std::move(file), 1.0,
                                                   static_cast<std::size_t>(input_rate / 100));
        }
        SessionOptions o;
        o.config = s.config;
        o.overrides = s.overrides;
        o.mode = RunMode::Live;
        o.live_io = &run->io;
        o.events = &view_;
        o.output_rate = kOutputRate;
        const std::string playing = from_file ? "Playing " + s.input_wav.filename().string() + "..." : "Listening...";
        o.on_ready = [&] {
            // The models are loaded: only now does the file start, as a speaker would.
            if (feeder) feeder->start();
            set_status(playing);
        };
        set_status("Loading the models...");
        auto session = std::make_unique<Session>(std::move(o));
        Session* active = session.get();
        {
            const std::lock_guard<std::mutex> lock(mu_);
            run->session = std::move(session);
        }
        if (run->stop.load()) active->stop();

        // The microphone and/or the speaker. Muted, the translation still drains at playback pace.
        AudioDevice::Config device_cfg;
        device_cfg.capture_device = s.capture_device;
        device_cfg.playback_device = s.playback_device;
        device_cfg.capture_rate = input_rate;
        device_cfg.playback_rate = kOutputRate;
        RingSource* capture = from_file ? nullptr : run->source.get();
        RingSink* playback = s.mute ? nullptr : run->sink.get();
        std::unique_ptr<AudioDevice> device;
        if (capture != nullptr || playback != nullptr) device = std::make_unique<AudioDevice>(device_cfg, capture, playback);
        std::unique_ptr<PacedDrain> drain;
        if (s.mute) drain = std::make_unique<PacedDrain>(*run->sink, 1.0, static_cast<std::size_t>(kOutputRate / 100));
        if (device) device->start();
        if (drain) drain->start();

        active->run();
        // Let what is still queued play out, unless stopped.
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
        while (!run->stop.load() && run->sink->queued() > 0 && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        if (feeder) feeder->stop();
        if (device) device->stop();
        if (drain) drain->stop();
        set_status(run->stop.load() ? "Stopped." : "Finished.");
    } catch (const std::exception& e) {
        set_status(std::string("Error: ") + e.what());
    }
    running_.store(false);
}

}  // namespace ee::desktop
