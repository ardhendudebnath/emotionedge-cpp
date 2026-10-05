// emotionedge: command-line front end for the EmotionEdge pipeline.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#endif

#include <nlohmann/json.hpp>

#include "core/audio/resampler.hpp"
#include "core/audio/wav.hpp"
#include "core/emotion/acoustic.hpp"
#include "core/emotion/fusion.hpp"
#include "core/emotion/lexical.hpp"
#include "core/emotion/prosody_features.hpp"
#include "core/pipeline/builtin_stages.hpp"
#include "core/pipeline/demo.hpp"
#include "core/pipeline/session.hpp"
#include "core/runtime/log.hpp"
#include "core/runtime/model_registry.hpp"
#include "core/telemetry/telemetry.hpp"
#include "core/translate/languages.hpp"
#include "core/tts/tts_engine.hpp"
#if defined(EE_HAVE_MINIAUDIO)
#include "core/audio/device.hpp"
#endif
#if defined(EE_HAVE_WEBSOCKET)
#include <thread>

#include "core/server/translation_server.hpp"
#endif

namespace fs = std::filesystem;
using namespace ee;

namespace {

constexpr const char* kUsage = R"(emotionedge - real-time, emotion-preserving speech translation

usage: emotionedge <command> [options]

commands:
  demo       Synthesize the blueprint walkthrough ("I can't believe you did this!", angry)
             and translate it end to end. --conversation: three utterances in three emotions.
  run        Translate a WAV file:  run --input speech.wav [--script words.json]
  live       Translate the microphone in real time (needs -DEE_WITH_MINIAUDIO=ON)
  devices    List audio devices (needs -DEE_WITH_MINIAUDIO=ON)
  serve      WebSocket server, one live session per connection (needs -DEE_WITH_WEBSOCKET=ON):
             serve [--port 8080] [--host 127.0.0.1] [--max-sessions 1]; see apps/README.md
  models     List or verify model files:  models [list|verify] [--manifest FILE]
  say        Speak text with a TTS engine into a WAV (test input for real ASR):
             say --text "One. | Two." [--engine piper --model-id ID --manifest FILE] [--out F]
  emotion    Run only the emotion models (the config's `emotion` stage) on audio and/or text:
             emotion --input a.wav [--text "..."]  |  emotion --manifest items.jsonl --out p.jsonl
  stages     List the registered stage types
  version    Print the version

options:
  --config FILE        pipeline graph (default: config/pipeline.yaml)
  --set KEY=VALUE      override a setting, e.g. --set translate.engine=ct2 (repeatable)
  --target LANG        target language, e.g. hi, es
  --out DIR            where outputs go (default: out/<command>)
  --realtime           threaded run with real-time pacing instead of offline
  --speed X            pacing speed for --realtime (default 1)
  --echo-sim DB        --realtime: feed what is played back into the input through a simulated
                       loudspeaker and room, DB relative to the playback (e.g. -6); also
                       --echo-delay-ms MS (40) and --echo-rt60-ms MS (250)
  --mix-source-db DB   also write mix.wav: the original under the translation, ducked
  --trace              write a Perfetto/Chrome trace (trace.json)
  --log LEVEL          debug | info | warn | error
)";

struct Args {
    std::string command;
    std::map<std::string, std::string> options;
    std::vector<std::pair<std::string, std::string>> sets;
    std::vector<std::string> positional;

    [[nodiscard]] bool has(const std::string& key) const { return options.count(key) > 0; }
    [[nodiscard]] std::string get(const std::string& key, const std::string& fallback = {}) const {
        const auto it = options.find(key);
        return it == options.end() ? fallback : it->second;
    }
};

const std::set<std::string> kFlags = {"realtime", "conversation", "help", "trace"};

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg.rfind("--", 0) != 0) {
            if (a.command.empty()) {
                a.command = arg;
            } else {
                a.positional.push_back(arg);
            }
            continue;
        }
        std::string key = arg.substr(2);
        std::string value;
        if (const auto eq = key.find('='); eq != std::string::npos) {
            value = key.substr(eq + 1);
            key = key.substr(0, eq);
        } else if (kFlags.count(key) > 0) {
            value = "1";
        } else if (i + 1 < argc) {
            value = argv[++i];
        } else {
            throw std::runtime_error("--" + key + " needs a value");
        }
        if (key == "set") {
            const auto eq = value.find('=');
            if (eq == std::string::npos) throw std::runtime_error("--set expects KEY=VALUE");
            a.sets.emplace_back(value.substr(0, eq), value.substr(eq + 1));
        } else {
            a.options[key] = value;
        }
    }
    return a;
}

fs::path resolve_config(const Args& args) {
    if (args.has("config")) return args.get("config");
    if (fs::exists("config/pipeline.yaml")) return "config/pipeline.yaml";
    return EE_DEFAULT_CONFIG;
}

std::string abs_path(const fs::path& p) { return fs::absolute(p).lexically_normal().generic_string(); }

std::string signed_value(float v, int decimals) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%+.*f", decimals, static_cast<double>(v));
    return buf;
}

std::string quoted_words(const std::string& text, const std::string& language, const std::vector<std::uint16_t>& idx) {
    const auto words = split_words(text, language);
    std::string out;
    for (std::uint16_t i : idx) {
        if (i >= words.size()) continue;
        if (!out.empty()) out += ", ";
        out += "\"" + words[i] + "\"";
    }
    return out.empty() ? "-" : out;
}

void print_report(const SessionResult& result, Session& session, bool realtime) {
    const PipelineSpec& spec = session.spec();
    std::printf("\nEmotionEdge-C++  %s -> %s  (%s)\n", spec.source_language.c_str(), spec.target_language.c_str(),
                realtime ? "real-time run" : "offline run");
    for (const UtteranceRecord& r : result.utterances) {
        if (r.source_text.empty() && r.translation.empty()) continue;
        std::printf("\nutterance %llu  [%.2f-%.2f s]\n", static_cast<unsigned long long>(r.id), r.src_start, r.src_end);
        std::printf("  1 HEARD       \"%s\"\n", r.source_text.c_str());
        std::printf("  2 FELT        %-8s V %s  A %s  D %s  confidence %.2f  emphasis %s\n",
                    std::string(to_string(r.emotion.label)).c_str(), signed_value(r.emotion.vad.v, 2).c_str(),
                    signed_value(r.emotion.vad.a, 2).c_str(), signed_value(r.emotion.vad.d, 2).c_str(),
                    static_cast<double>(r.emotion.confidence),
                    quoted_words(r.source_text, r.source_language, r.source_emphasis).c_str());
        std::printf("  3 MT INPUT    %s\n", r.mt_input.c_str());
        std::printf("  4 TRANSLATED  %s   emphasis %s\n", r.translation.c_str(),
                    quoted_words(r.translation, r.target_language, r.target_emphasis).c_str());
        const ProsodyTargets& p = r.prosody;
        std::printf("  5 PROSODY     pitch %s%%  range %s%%  rate %s%%  energy %s dB  pause %.0f ms  accent x%.2f\n",
                    signed_value(p.pitch_pct, 0).c_str(), signed_value(p.range_pct, 0).c_str(),
                    signed_value(p.rate_pct, 0).c_str(), signed_value(p.energy_db, 1).c_str(),
                    static_cast<double>(p.pause_ms), static_cast<double>(p.accent));
        if (!r.ecs.empty()) {
            std::printf("  6 ECS         %.2f (target >= 0.75, %zu clause%s)\n", static_cast<double>(r.ecs_mean()),
                        r.ecs.size(), r.ecs.size() == 1 ? "" : "s");
        }
        if (r.out_start >= 0.0) std::printf("    output      %.2f-%.2f s\n", r.out_start, r.out_end);
    }

    const auto& t = session.telemetry();
    std::printf("\nlatency after speech end%s\n", realtime ? "" : " (offline: processing time only, no real-time waits)");
    std::printf("  %-26s %9s %9s %9s\n", "stage", "p50 ms", "p95 ms", "budget");
    for (const auto& row : t.budget_report()) {
        if (row.snapshot.count == 0) continue;
        std::printf("  %-26s %9.1f %9.1f %9.0f\n", std::string(row.row->name).c_str(),
                    static_cast<double>(row.snapshot.percentile(0.5)) / 1000.0,
                    static_cast<double>(row.snapshot.percentile(0.95)) / 1000.0,
                    static_cast<double>(row.row->target_us) / 1000.0);
    }
    const auto e2e = t.end_to_end();
    if (e2e.count > 0) {
        std::printf("  %-26s %9.1f %9.1f %9.0f\n", "end to end (p95 target)", static_cast<double>(e2e.percentile(0.5)) / 1000.0,
                    static_cast<double>(e2e.percentile(0.95)) / 1000.0, 800.0);
    }
}

SessionOptions base_options(const Args& args, const fs::path& out) {
    SessionOptions opts;
    opts.config = resolve_config(args);
    opts.mode = args.has("realtime") ? RunMode::Realtime : RunMode::Offline;
    opts.speed = std::stod(args.get("speed", "1"));
    opts.overrides.emplace_back("recorder.srt", abs_path(out / "captions.srt"));
    opts.overrides.emplace_back("recorder.json", abs_path(out / "session.json"));
    opts.overrides.emplace_back("telemetry.prometheus", abs_path(out / "metrics.prom"));
    if (args.has("trace")) opts.overrides.emplace_back("telemetry.trace", abs_path(out / "trace.json"));
    if (args.has("target")) opts.overrides.emplace_back("pipeline.target_language", args.get("target"));
    if (args.has("echo-sim")) {
        if (opts.mode != RunMode::Realtime) throw std::runtime_error("--echo-sim needs --realtime");
        EchoSimulator::Config room;
        room.gain_db = std::stof(args.get("echo-sim"));
        if (args.has("echo-delay-ms")) room.delay_ms = std::stof(args.get("echo-delay-ms"));
        if (args.has("echo-rt60-ms")) room.rt60_ms = std::stof(args.get("echo-rt60-ms"));
        opts.echo_sim = room;
    }
    for (const auto& kv : args.sets) opts.overrides.push_back(kv);
    return opts;
}

void write_outputs(const Args& args, const SessionResult& result, const std::vector<float>& input, int input_rate,
                   const fs::path& out) {
    write_wav(out / "translated.wav", result.output_audio, result.output_rate);
    std::printf("\nwrote %s\n", abs_path(out / "translated.wav").c_str());
    if (args.has("mix-source-db")) {
        const auto mix = mix_with_ducking(result.output_audio, input, input_rate, result.output_rate,
                                          std::stof(args.get("mix-source-db")), -12.0f);
        write_wav(out / "mix.wav", mix, result.output_rate);
        std::printf("wrote %s\n", abs_path(out / "mix.wav").c_str());
    }
    for (const char* name : {"captions.srt", "session.json", "metrics.prom", "trace.json"}) {
        if (fs::exists(out / name)) std::printf("wrote %s\n", abs_path(out / name).c_str());
    }
}

int cmd_demo(const Args& args) {
    const fs::path out = args.get("out", "out/demo");
    fs::create_directories(out);
    const DemoInput demo = args.has("conversation") ? make_demo_conversation() : make_walkthrough_input();
    write_wav(out / "input.wav", demo.audio, demo.sample_rate);
    {
        std::ofstream script(out / "input.script.json", std::ios::binary);
        script << demo.script_json;
    }
    SessionOptions opts = base_options(args, out);
    opts.overrides.insert(opts.overrides.begin(), {{"asr.engine", "scripted"}, {"asr.script", abs_path(out / "input.script.json")}});
    opts.input_samples = demo.audio;
    opts.input_rate = demo.sample_rate;
    Session session(opts);
    const SessionResult result = session.run();
    print_report(result, session, opts.mode == RunMode::Realtime);
    write_outputs(args, result, demo.audio, demo.sample_rate, out);
    std::printf("wrote %s (synthetic source speech)\n", abs_path(out / "input.wav").c_str());
    return result.completed ? 0 : 1;
}

int cmd_run(const Args& args) {
    if (!args.has("input")) throw std::runtime_error("run needs --input FILE.wav");
    const fs::path out = args.get("out", "out/run");
    fs::create_directories(out);
    const WavData wav = read_wav(args.get("input"));
    SessionOptions opts = base_options(args, out);
    if (args.has("script")) {
        opts.overrides.insert(opts.overrides.begin(), {{"asr.engine", "scripted"}, {"asr.script", abs_path(args.get("script"))}});
    }
    opts.input_samples = wav.samples;
    opts.input_rate = wav.sample_rate;
    Session session(opts);
    const SessionResult result = session.run();
    print_report(result, session, opts.mode == RunMode::Realtime);
    write_outputs(args, result, wav.samples, wav.sample_rate, out);
    return result.completed ? 0 : 1;
}

#if defined(EE_HAVE_MINIAUDIO)
std::atomic<Session*> g_session{nullptr};

void on_interrupt(int) {
    if (Session* s = g_session.load()) s->stop();  // only an atomic store: async-signal-safe
}
#endif

int cmd_live(const Args& args) {
#if defined(EE_HAVE_MINIAUDIO)
    const fs::path out = args.get("out", "out/live");
    fs::create_directories(out);
    RingSource capture(16000, 16000 * 10);
    RingSink playback(24000, 24000 * 10);
    SpscRing<float> echo(24000 * 2);  // what the speaker plays: the echo canceller's reference
    playback.set_played_tap(&echo);
    AudioIo io;
    io.capture = &capture;
    io.playback = &playback;
    io.echo_reference = &echo;
    AudioDevice::Config device_cfg;
    device_cfg.capture_device = args.get("capture-device");
    device_cfg.playback_device = args.get("playback-device");
    AudioDevice device(device_cfg, &capture, &playback);

    SessionOptions opts = base_options(args, out);
    opts.mode = RunMode::Live;
    opts.live_io = &io;
    Session session(opts);
    g_session.store(&session);
    std::signal(SIGINT, on_interrupt);
    device.start();
    std::printf("listening... press Ctrl+C to stop\n");
    const SessionResult result = session.run();
    device.stop();
    g_session.store(nullptr);
    print_report(result, session, true);
    return 0;
#else
    (void)args;
    throw std::runtime_error("live needs a build with -DEE_WITH_MINIAUDIO=ON");
#endif
}

#if defined(EE_HAVE_WEBSOCKET)
std::atomic<bool> g_serve_stop{false};

void on_serve_interrupt(int) { g_serve_stop.store(true); }  // lock-free atomic: async-signal-safe
#endif

int cmd_serve(const Args& args) {
#if defined(EE_HAVE_WEBSOCKET)
    TranslationServer::Options o;
    o.host = args.get("host", "127.0.0.1");
    o.port = std::stoi(args.get("port", "8080"));
    o.max_sessions = std::stoul(args.get("max-sessions", "1"));
    o.config = resolve_config(args);
    if (args.has("target")) o.overrides.emplace_back("pipeline.target_language", args.get("target"));
    for (const auto& kv : args.sets) o.overrides.push_back(kv);
    TranslationServer server(o);
    server.start();
    std::printf("serving ws://%s:%d/ with %s: one live session per connection, up to %zu at a time. "
                "Ctrl+C stops.\n",
                o.host.c_str(), o.port, o.config.generic_string().c_str(), o.max_sessions);
    std::signal(SIGINT, on_serve_interrupt);
    std::signal(SIGTERM, on_serve_interrupt);
    while (!g_serve_stop.load()) std::this_thread::sleep_for(std::chrono::milliseconds(200));
    server.stop();
    return 0;
#else
    (void)args;
    throw std::runtime_error("serve needs a build with -DEE_WITH_WEBSOCKET=ON");
#endif
}

int cmd_devices() {
#if defined(EE_HAVE_MINIAUDIO)
    for (const AudioDeviceInfo& d : list_audio_devices()) {
        std::printf("%-8s %s%s\n", d.capture ? "capture" : "playback", d.name.c_str(), d.is_default ? "  (default)" : "");
    }
    return 0;
#else
    throw std::runtime_error("devices needs a build with -DEE_WITH_MINIAUDIO=ON");
#endif
}

int cmd_models(const Args& args) {
    fs::path manifest = args.get("manifest");
    if (manifest.empty()) {
        const PipelineSpec spec = load_pipeline(resolve_config(args));
        manifest = spec.models_manifest.empty() ? fs::path("models/manifest.json") : fs::path(spec.models_manifest);
    }
    const ModelRegistry registry = ModelRegistry::load(manifest);
    const bool verify = !args.positional.empty() && args.positional[0] == "verify";
    int problems = 0;
    for (const ModelEntry& m : registry.models()) {
        std::string status = "not checked";
        if (verify) {
            const VerifyResult r = registry.verify(m);
            status = std::string(to_string(r.status));
            if (r.status == VerifyStatus::Unpinned) status += " (sha256 " + r.actual_sha256 + ")";
            if (r.status != VerifyStatus::Ok && r.status != VerifyStatus::Unpinned) ++problems;
        } else {
            status = fs::exists(m.check_file.empty() ? m.resolved : m.resolved / m.check_file) ? "present" : "missing";
        }
        std::printf("%-34s %-6s %-6s %s\n", m.id.c_str(), m.task.c_str(), m.format.c_str(), status.c_str());
    }
    return problems == 0 ? 0 : 1;
}

int cmd_stages() {
    for (const std::string& type : builtin_registry().types()) std::printf("%s\n", type.c_str());
    return 0;
}

/// Speaks text with a TTS engine, e.g. to make test input for the real ASR:
///   say --text "First sentence. | Second sentence." --engine piper --model-id tts.piper.en_US.lessac.medium
int cmd_say(const Args& args) {
    if (!args.has("text")) throw std::runtime_error("say needs --text \"...\" (use | between utterances)");
    Params params;
    params.set("engine", args.get("engine", "formant"));
    for (const char* key : {"model", "model_id", "espeak_data", "voice_config", "speaker"}) {
        std::string opt(key);
        std::replace(opt.begin(), opt.end(), '_', '-');
        if (args.has(opt)) params.set(key, args.get(opt));
    }
    // Any other engine parameter: --set tts.device=cuda, --set tts.voice=hm_omega, ...
    for (const auto& [key, value] : args.sets) {
        if (key.rfind("tts.", 0) != 0) throw std::runtime_error("say takes only tts.<param> overrides, not '" + key + "'");
        params.set(key.substr(4), value);
    }
    std::unique_ptr<ModelRegistry> registry;
    if (const fs::path manifest = args.get("manifest"); !manifest.empty()) {
        registry = std::make_unique<ModelRegistry>(ModelRegistry::load(manifest));
    }
    auto engine = make_tts_engine(params, registry.get());
    const double gap = std::stod(args.get("gap", "1.2"));
    const std::string language = args.get("language", "en");

    std::vector<float> audio(static_cast<std::size_t>(0.5 * engine->sample_rate()), 0.0f);
    SynthesisResult result;
    std::string text = args.get("text");
    std::size_t start = 0;
    while (start <= text.size()) {
        const std::size_t bar = text.find('|', start);
        std::string sentence = text.substr(start, bar == std::string::npos ? std::string::npos : bar - start);
        start = bar == std::string::npos ? text.size() + 1 : bar + 1;
        sentence.erase(0, sentence.find_first_not_of(' '));
        sentence.erase(sentence.find_last_not_of(' ') + 1);
        if (sentence.empty()) continue;
        SynthesisRequest req;
        req.text = sentence;
        req.language = language;
        engine->synthesize(req, result);
        audio.insert(audio.end(), result.audio.begin(), result.audio.end());
        audio.insert(audio.end(), static_cast<std::size_t>(gap * engine->sample_rate()), 0.0f);
    }
    const fs::path out = args.get("out", "out/say.wav");
    if (out.has_parent_path()) fs::create_directories(out.parent_path());
    write_wav(out, audio, engine->sample_rate());
    std::printf("wrote %s (%.1f s at %d Hz)\n", abs_path(out).c_str(),
                static_cast<double>(audio.size()) / engine->sample_rate(), engine->sample_rate());
    return 0;
}

/// Runs only the emotion models of the pipeline's `emotion` stage (same params, models and fusion
/// as a full run) on WAV files and/or text, e.g. to evaluate them on a labelled set:
///   emotion --input a.wav [--text "..."]           one item, printed as JSON
///   emotion --manifest items.jsonl --out p.jsonl   {"id", "audio", "text"} per line -> predictions
int cmd_emotion(const Args& args) {
    PipelineSpec spec = load_pipeline(resolve_config(args));
    for (const auto& [key, value] : args.sets) apply_override(spec, key, value);
    spec.apply_device_default();
    const StageSpec* stage = spec.find_stage("emotion");
    if (stage == nullptr) throw std::runtime_error("the pipeline has no 'emotion' stage");
    const Params& params = stage->params;
    std::unique_ptr<ModelRegistry> registry;
    if (!spec.models_manifest.empty() && fs::exists(spec.models_manifest)) {
        registry = std::make_unique<ModelRegistry>(ModelRegistry::load(spec.models_manifest));
    }
    const auto acoustic = make_acoustic_model(params, registry.get());
    const auto lexical = make_lexical_model(params, registry.get());
    const FusionConfig fusion = FusionConfig::from(params);
    const std::string language = args.get("language", spec.source_language);

    std::vector<nlohmann::json> items;
    if (args.has("manifest")) {
        std::ifstream in(args.get("manifest"));
        if (!in) throw std::runtime_error("cannot open " + args.get("manifest"));
        for (std::string line; std::getline(in, line);) {
            if (line.find_first_not_of(" \t\r") != std::string::npos) items.push_back(nlohmann::json::parse(line));
        }
    } else if (args.has("input") || args.has("text")) {
        items.push_back({{"id", args.get("input", "text")}, {"audio", args.get("input")}, {"text", args.get("text")}});
    } else {
        throw std::runtime_error("emotion needs --input FILE.wav and/or --text \"...\", or --manifest FILE.jsonl");
    }

    const auto modality = [](const ModalityEstimate& e, double ms) {
        return nlohmann::json{{"valid", e.valid},
                              {"vad", {e.vad.v, e.vad.a, e.vad.d}},
                              {"confidence", {e.confidence.v, e.confidence.a, e.confidence.d}},
                              {"ms", ms}};
    };
    const auto elapsed_ms = [](std::chrono::steady_clock::time_point start) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    };
    std::ofstream out_file;
    if (args.has("out")) {
        const fs::path out = args.get("out");
        if (out.has_parent_path()) fs::create_directories(out.parent_path());
        out_file.open(out);
    }
    for (const nlohmann::json& item : items) {
        ModalityEstimate a;
        ModalityEstimate l;
        double a_ms = 0.0;
        double l_ms = 0.0;
        if (const std::string audio = item.value("audio", std::string()); !audio.empty()) {
            const WavData wav = read_wav(audio);
            const std::vector<float> samples = Resampler::convert(wav.samples, wav.sample_rate, 16000);
            ProsodyTracker tracker(16000);
            tracker.push(samples);
            tracker.flush();
            const auto start = std::chrono::steady_clock::now();
            a = acoustic->estimate(samples, tracker.frames(), tracker.hop_seconds());
            a_ms = elapsed_ms(start);
        }
        if (const std::string text = item.value("text", std::string()); !text.empty() && lexical != nullptr) {
            const auto start = std::chrono::steady_clock::now();
            l = lexical->estimate(text, language);
            l_ms = elapsed_ms(start);
        }
        const EmotionState fused = fuse(a, l, fusion);
        nlohmann::json row{{"id", item.value("id", std::string())},
                           {"acoustic", modality(a, a_ms)},
                           {"lexical", modality(l, l_ms)},
                           {"fused",
                            {{"vad", {fused.vad.v, fused.vad.a, fused.vad.d}},
                             {"label", std::string(to_string(fused.label))},
                             {"confidence", fused.confidence}}}};
        if (out_file.is_open()) {
            out_file << row.dump() << '\n';
        } else {
            std::printf("%s\n", row.dump(2).c_str());
        }
    }
    if (out_file.is_open()) std::printf("wrote %zu predictions to %s\n", items.size(), abs_path(args.get("out")).c_str());
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
#if defined(_WIN32)
    SetConsoleOutputCP(CP_UTF8);
#endif
    try {
        const Args args = parse_args(argc, argv);
        if (args.has("log")) {
            const auto level = log::parse_level(args.get("log"));
            if (!level) throw std::runtime_error("unknown log level '" + args.get("log") + "'");
            log::set_level(*level);
        }
        if (args.command.empty() || args.command == "help" || args.has("help")) {
            std::printf("%s", kUsage);
            return args.command.empty() ? 1 : 0;
        }
        if (args.command == "demo") return cmd_demo(args);
        if (args.command == "run") return cmd_run(args);
        if (args.command == "live") return cmd_live(args);
        if (args.command == "devices") return cmd_devices();
        if (args.command == "serve") return cmd_serve(args);
        if (args.command == "models") return cmd_models(args);
        if (args.command == "stages") return cmd_stages();
        if (args.command == "say") return cmd_say(args);
        if (args.command == "emotion") return cmd_emotion(args);
        if (args.command == "version") {
            std::printf("emotionedge %s\n", EE_VERSION);
            return 0;
        }
        std::fprintf(stderr, "unknown command '%s'\n\n%s", args.command.c_str(), kUsage);
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
