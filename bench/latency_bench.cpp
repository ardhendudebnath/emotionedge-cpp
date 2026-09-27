// ee_bench: runs the pipeline over a synthetic conversation, paced in real time, and checks the
// blueprint's latency budget (p.2) and success targets. With the stand-in engines it measures
// the runtime's own overhead; with pipeline.engines.yaml it measures the real engines.
//
//   ee_bench [--config FILE] [--repeat N] [--speed X] [--offline] [--set KEY=VALUE]
//            [--target LANG] [--out DIR] [--json FILE] [--strict]
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#endif

#include "core/pipeline/demo.hpp"
#include "core/pipeline/session.hpp"
#include "core/telemetry/telemetry.hpp"

namespace fs = std::filesystem;
using json = nlohmann::json;
using namespace ee;

namespace {

double peak_rss_mb() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof pmc)) return static_cast<double>(pmc.PeakWorkingSetSize) / 1048576.0;
    return 0.0;
#else
    rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
#if defined(__APPLE__)
    return static_cast<double>(usage.ru_maxrss) / 1048576.0;  // bytes
#else
    return static_cast<double>(usage.ru_maxrss) / 1024.0;  // kilobytes
#endif
#endif
}

double ms(std::uint64_t us) { return static_cast<double>(us) / 1000.0; }

}  // namespace

int main(int argc, char** argv) {
    try {
        std::map<std::string, std::string> opt;
        std::vector<std::pair<std::string, std::string>> sets;
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            if (a == "--offline" || a == "--strict") {
                opt[a.substr(2)] = "1";
            } else if (a == "--set" && i + 1 < argc) {
                const std::string kv = argv[++i];
                const auto eq = kv.find('=');
                if (eq == std::string::npos) throw std::runtime_error("--set expects KEY=VALUE");
                sets.emplace_back(kv.substr(0, eq), kv.substr(eq + 1));
            } else if (a.rfind("--", 0) == 0 && i + 1 < argc) {
                opt[a.substr(2)] = argv[++i];
            } else {
                throw std::runtime_error("unknown argument '" + a + "'");
            }
        }
        const auto get = [&](const std::string& k, const std::string& fallback) {
            const auto it = opt.find(k);
            return it == opt.end() ? fallback : it->second;
        };
        const int repeat = std::max(1, std::stoi(get("repeat", "2")));
        const bool offline = opt.count("offline") > 0;
        const fs::path out = get("out", "out/bench");
        fs::create_directories(out);

        std::vector<DemoUtterance> utterances;
        for (int r = 0; r < repeat; ++r) {
            for (const DemoUtterance& u : demo_conversation_utterances()) utterances.push_back(u);
        }
        const DemoInput input = render_demo(utterances);
        const fs::path script = fs::absolute(out / "input.script.json");
        std::ofstream(script, std::ios::binary) << input.script_json;

        SessionOptions opts;
        opts.config = opt.count("config") ? fs::path(get("config", "")) : fs::path(EE_DEFAULT_CONFIG);
        if (!opt.count("config") && fs::exists("config/pipeline.yaml")) opts.config = "config/pipeline.yaml";
        opts.mode = offline ? RunMode::Offline : RunMode::Realtime;
        opts.speed = std::stod(get("speed", "1"));
        opts.input_samples = input.audio;
        opts.input_rate = input.sample_rate;
        opts.overrides = {{"asr.engine", "scripted"},
                          {"asr.script", script.generic_string()},
                          {"recorder.json", fs::absolute(out / "session.json").generic_string()}};
        if (opt.count("target")) opts.overrides.emplace_back("pipeline.target_language", get("target", ""));
        for (const auto& kv : sets) opts.overrides.push_back(kv);

        std::printf("ee_bench: %zu utterances, %.1f s of speech, %s\n", utterances.size(),
                    static_cast<double>(input.audio.size()) / input.sample_rate,
                    offline ? "offline (processing time only)" : "real-time paced");
        Session session(opts);
        const SessionResult result = session.run();
        const auto& t = session.telemetry();

        std::printf("\n%-26s %9s %9s %9s\n", "latency budget (p.2)", "p50 ms", "p95 ms", "target");
        json rows = json::array();
        for (const auto& row : t.budget_report()) {
            const double p50 = ms(row.snapshot.percentile(0.5));
            const double p95 = ms(row.snapshot.percentile(0.95));
            std::printf("%-26s %9.1f %9.1f %9.0f%s\n", std::string(row.row->name).c_str(), p50, p95,
                        ms(row.row->target_us), row.snapshot.count == 0 ? "  (no data)" : "");
            rows.push_back({{"row", row.row->name}, {"target_ms", ms(row.row->target_us)}, {"p50_ms", p50},
                            {"p95_ms", p95}, {"count", row.snapshot.count}});
        }

        const auto e2e = t.end_to_end();
        const auto rtf = t.rtf();
        const auto ecs = t.ecs();
        const double e2e_p95 = ms(e2e.percentile(0.95));
        const double rtf_p95 = static_cast<double>(rtf.percentile(0.95)) / 10000.0;
        const double ecs_mean = ecs.mean() / 10000.0;
        const double rss = peak_rss_mb();
        const std::uint64_t dropouts = t.dropouts();

        struct Target {
            const char* name;
            std::string value;
            const char* goal;
            bool met;
        };
        char buf[4][64];
        std::snprintf(buf[0], sizeof buf[0], "%.1f ms", e2e_p95);
        std::snprintf(buf[1], sizeof buf[1], "%.3f", rtf_p95);
        std::snprintf(buf[2], sizeof buf[2], "%.2f", ecs_mean);
        std::snprintf(buf[3], sizeof buf[3], "%.0f MB", rss);
        const bool have_e2e = e2e.count > 0;
        const std::vector<Target> targets = {
            {"End-to-end latency, p95", buf[0], "< 800 ms", have_e2e && e2e_p95 < 800.0},
            {"ASR real-time factor, p95", buf[1], "< 0.3", rtf.count > 0 && rtf_p95 < 0.3},
            {"Emotion consistency (ECS)", buf[2], ">= 0.75", ecs.count > 0 && ecs_mean >= 0.75},
            {"Peak RAM", buf[3], "< 3 GB", rss < 3072.0},
            {"Audio dropouts", std::to_string(dropouts) + " samples", "0 / hour", dropouts == 0},
        };
        std::printf("\n%-28s %14s %10s\n", "success targets (p.2)", "measured", "target");
        bool all = true;
        for (const Target& target : targets) {
            std::printf("%-28s %14s %10s  %s\n", target.name, target.value.c_str(), target.goal, target.met ? "ok" : "MISSED");
            all = all && target.met;
        }
        std::printf("\n%zu utterances completed, %llu frames shed under backpressure\n", result.utterances.size(),
                    static_cast<unsigned long long>(result.dropped_frames));

        json report = {{"mode", offline ? "offline" : "realtime"},
                       {"config", opts.config.generic_string()},
                       {"utterances", result.utterances.size()},
                       {"end_to_end_ms", {{"p50", ms(e2e.percentile(0.5))}, {"p95", e2e_p95}, {"count", e2e.count}}},
                       {"rows", rows},
                       {"asr_rtf", {{"p50", static_cast<double>(rtf.percentile(0.5)) / 10000.0}, {"p95", rtf_p95}}},
                       {"ecs", {{"mean", ecs_mean}, {"count", ecs.count}}},
                       {"peak_rss_mb", rss},
                       {"dropout_samples", dropouts},
                       {"targets_met", all}};
        const fs::path report_path = get("json", (out / "bench.json").string());
        std::ofstream(report_path) << report.dump(2) << "\n";
        std::printf("wrote %s\n", fs::absolute(report_path).generic_string().c_str());
        return opt.count("strict") > 0 && !all ? 1 : 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
