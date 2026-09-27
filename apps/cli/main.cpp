// emotionedge: command-line front end for the EmotionEdge pipeline.
#include <atomic>
#include <csignal>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#endif

#include "core/audio/wav.hpp"
#include "core/pipeline/builtin_stages.hpp"
#include "core/pipeline/demo.hpp"
#include "core/pipeline/session.hpp"
#include "core/runtime/log.hpp"
#include "core/runtime/model_registry.hpp"
#include "core/telemetry/telemetry.hpp"
#include "core/translate/languages.hpp"
#if defined(EE_HAVE_MINIAUDIO)
#include "core/audio/device.hpp"
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
  models     List or verify model files:  models [list|verify] [--manifest FILE]
  stages     List the registered stage types
  version    Print the version

options:
  --config FILE        pipeline graph (default: config/pipeline.yaml)
  --set KEY=VALUE      override a setting, e.g. --set translate.engine=ct2 (repeatable)
  --target LANG        target language, e.g. hi, es
  --out DIR            where outputs go (default: out/<command>)
  --realtime           threaded run with real-time pacing instead of offline
  --speed X            pacing speed for --realtime (default 1)
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
    SpscRing<float> echo(32000);
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
        if (args.command == "models") return cmd_models(args);
        if (args.command == "stages") return cmd_stages();
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
