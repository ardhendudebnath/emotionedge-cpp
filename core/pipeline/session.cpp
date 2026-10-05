#include "core/pipeline/session.hpp"

#include <algorithm>
#include <cmath>
#include <thread>

#include "core/audio/dsp.hpp"
#include "core/audio/resampler.hpp"
#include "core/audio/wav.hpp"
#include "core/pipeline/builtin_stages.hpp"
#include "core/runtime/graph.hpp"
#include "core/runtime/log.hpp"
#include "core/runtime/model_registry.hpp"

namespace ee {

namespace {

std::vector<UtteranceRecord> collect_records(Graph& graph, const PipelineSpec& spec) {
    for (const StageSpec& s : spec.stages) {
        if (s.type != "recorder") continue;
        if (auto* recorder = dynamic_cast<RecorderStage*>(graph.stage(s.name))) return recorder->records();
    }
    return {};
}

}  // namespace

Session::Session(SessionOptions options)
    : options_(std::move(options)), telemetry_(std::make_unique<telemetry::Telemetry>()) {
    spec_ = load_pipeline(options_.config);
    for (const auto& [key, value] : options_.overrides) apply_override(spec_, key, value);
    validate(spec_);
}

Session::~Session() = default;

SessionResult Session::run() {
    SessionResult result;

    std::unique_ptr<ModelRegistry> models;
    if (!spec_.models_manifest.empty()) {
        if (std::filesystem::exists(spec_.models_manifest)) {
            models = std::make_unique<ModelRegistry>(ModelRegistry::load(spec_.models_manifest));
        } else {
            log::warn("model manifest '", spec_.models_manifest, "' not found; stages must name model files directly");
        }
    }

    std::vector<float> input = options_.input_samples;
    int input_rate = options_.input_rate;
    if (!options_.input_wav.empty()) {
        WavData wav = read_wav(options_.input_wav);
        input = std::move(wav.samples);
        input_rate = wav.sample_rate;
    }
    if (options_.mode != RunMode::Live && (input.empty() || input_rate <= 0)) {
        throw ConfigError("a session needs input audio (a WAV file or samples)");
    }
    if (input_rate > 0) result.input_seconds = static_cast<double>(input.size()) / input_rate;

    Services services;
    services.telemetry = telemetry_.get();
    services.models = models.get();
    services.events = options_.events;
    const StageRegistry registry = builtin_registry();

    switch (options_.mode) {
    case RunMode::Offline: {
        BufferSource source(input, input_rate);
        TimelineSink timeline(options_.output_rate);
        AudioIo io;
        io.capture = &source;
        io.timeline = &timeline;
        services.audio = &io;
        Graph graph(spec_, registry, services);
        result.completed = graph.run_deterministic();
        result.utterances = collect_records(graph, spec_);
        result.dropped_frames = graph.dropped_frames();
        result.output_audio = timeline.samples();
        result.output_rate = options_.output_rate;
        break;
    }
    case RunMode::Realtime: {
        RingSource source(input_rate, static_cast<std::size_t>(input_rate) * 10);
        RingSink sink(options_.output_rate, static_cast<std::size_t>(options_.output_rate) * 30);
        SpscRing<float> echo(static_cast<std::size_t>(options_.output_rate) * 2);
        sink.set_played_tap(&echo);
        AudioIo io;
        io.capture = &source;
        io.playback = &sink;
        io.echo_reference = &echo;
        services.audio = &io;
        Graph graph(spec_, registry, services);
        PacedFeeder feeder(source, input, options_.speed, static_cast<std::size_t>(input_rate / 100));
        PacedDrain drain(sink, options_.speed, static_cast<std::size_t>(options_.output_rate / 100));
        std::unique_ptr<EchoSimulator> room;
        if (options_.echo_sim) {
            room = std::make_unique<EchoSimulator>(*options_.echo_sim, options_.output_rate, input_rate);
            drain.set_play_hook([r = room.get()](std::span<const float> played) { r->played(played); });
            feeder.set_capture_hook([r = room.get()](std::span<float> captured) { r->add_to(captured); });
        }
        graph.start();
        drain.start();
        feeder.start();

        const auto guard = options_.timeout.count() > 0
                               ? options_.timeout
                               : std::chrono::milliseconds(static_cast<long long>(result.input_seconds / options_.speed * 1500.0) + 15000);
        const TimePoint deadline = Clock::now() + guard;
        while (!graph.finished() && !stop_requested_.load() && Clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        result.completed = graph.finished();
        // Let the "speaker" play out what is still queued.
        const TimePoint drain_deadline = Clock::now() + std::chrono::seconds(10);
        while (sink.queued() > 0 && !stop_requested_.load() && Clock::now() < drain_deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        feeder.stop();
        graph.stop();
        drain.stop();
        telemetry_->count_dropout(source.overflow_samples() + sink.underrun_samples());
        result.utterances = collect_records(graph, spec_);
        result.dropped_frames = graph.dropped_frames();
        result.output_audio = drain.recording();
        result.output_rate = options_.output_rate;
        break;
    }
    case RunMode::Live: {
        if (options_.live_io == nullptr) throw ConfigError("live mode needs audio devices (AudioIo)");
        services.audio = options_.live_io;
        Graph graph(spec_, registry, services);
        graph.start();
        if (options_.on_ready) options_.on_ready();
        const TimePoint deadline = Clock::now() + (options_.timeout.count() > 0 ? options_.timeout : std::chrono::hours(24 * 365));
        while (!graph.finished() && !stop_requested_.load() && Clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        graph.stop();
        result.completed = true;
        result.utterances = collect_records(graph, spec_);
        result.dropped_frames = graph.dropped_frames();
        break;
    }
    }

    if (options_.mode != RunMode::Live && !input.empty()) {
        // The output covers at least the input's duration, so the files line up.
        const auto min_len = static_cast<std::size_t>(result.input_seconds * result.output_rate);
        if (result.output_audio.size() < min_len) result.output_audio.resize(min_len, 0.0f);
        if (options_.source_mix_db > -100.0f) {
            result.output_audio = mix_with_ducking(result.output_audio, input, input_rate, result.output_rate,
                                                   options_.source_mix_db, options_.duck_db);
        }
        if (!options_.output_wav.empty()) write_wav(options_.output_wav, result.output_audio, result.output_rate);
    }
    return result;
}

std::vector<float> mix_with_ducking(const std::vector<float>& translation, const std::vector<float>& source,
                                    int source_rate, int rate, float source_db, float duck_db) {
    const std::vector<float> src = Resampler::convert(source, source_rate, rate);
    std::vector<float> out(std::max(translation.size(), src.size()), 0.0f);
    const float source_gain = db_to_gain(source_db);
    const float duck_gain = db_to_gain(duck_db);
    const double attack = 1.0 - std::exp(-1.0 / (0.01 * rate));
    const double release = 1.0 - std::exp(-1.0 / (0.3 * rate));
    const double smooth = 1.0 - std::exp(-1.0 / (0.05 * rate));
    double envelope = 0.0;
    double duck = 1.0;
    for (std::size_t i = 0; i < out.size(); ++i) {
        const double t = i < translation.size() ? translation[i] : 0.0;
        const double level = std::abs(t);
        envelope += (level > envelope ? attack : release) * (level - envelope);
        const double target = envelope > 0.003 ? duck_gain : 1.0;  // translation audible above ~-50 dBFS
        duck += smooth * (target - duck);
        const double s = i < src.size() ? src[i] : 0.0;
        out[i] = soft_limit(static_cast<float>(t + s * source_gain * duck), 0.95f);
    }
    return out;
}

}  // namespace ee
