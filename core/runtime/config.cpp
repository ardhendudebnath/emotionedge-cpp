#include "core/runtime/config.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <fstream>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <unordered_map>

namespace ee {

namespace {

std::string expand_vars(std::string value, const std::string& config_dir) {
    static constexpr std::string_view kVar = "${config_dir}";
    for (std::size_t pos = value.find(kVar); pos != std::string::npos;
         pos = value.find(kVar, pos + config_dir.size())) {
        value.replace(pos, kVar.size(), config_dir);
    }
    return value;
}

void flatten(const YAML::Node& node, const std::string& prefix, Params& out,
             const std::string& config_dir) {
    switch (node.Type()) {
    case YAML::NodeType::Scalar: out.set(prefix, expand_vars(node.Scalar(), config_dir)); break;
    case YAML::NodeType::Sequence:
        for (std::size_t i = 0; i < node.size(); ++i) {
            flatten(node[i], prefix + "." + std::to_string(i), out, config_dir);
        }
        break;
    case YAML::NodeType::Map:
        for (const auto& kv : node) {
            const std::string key = kv.first.as<std::string>();
            flatten(kv.second, prefix.empty() ? key : prefix + "." + key, out, config_dir);
        }
        break;
    case YAML::NodeType::Null:
    case YAML::NodeType::Undefined: out.set(prefix, ""); break;
    }
}

std::string scalar(const YAML::Node& node, std::string_view what, const std::string& config_dir) {
    if (!node || !node.IsScalar()) throw ConfigError("missing or non-scalar '" + std::string(what) + "'");
    return expand_vars(node.Scalar(), config_dir);
}

template <typename T>
T scalar_as(const YAML::Node& node, std::string_view what) {
    try {
        return node.as<T>();
    } catch (const YAML::Exception&) {
        throw ConfigError("invalid value for '" + std::string(what) + "'");
    }
}

std::vector<std::string> string_list(const YAML::Node& node, const std::string& config_dir) {
    std::vector<std::string> out;
    if (!node) return out;
    if (node.IsScalar()) {
        out.push_back(expand_vars(node.Scalar(), config_dir));
    } else if (node.IsSequence()) {
        for (const auto& item : node) out.push_back(expand_vars(item.as<std::string>(), config_dir));
    } else {
        throw ConfigError("expected a string or a list of strings");
    }
    return out;
}

std::uint32_t parse_kinds(const YAML::Node& node) {
    if (!node) return kAllKinds;
    std::uint32_t mask = 0;
    for (const std::string& name : string_list(node, {})) {
        const auto kind = parse_frame_kind(name);
        if (!kind) throw ConfigError("unknown frame kind '" + name + "' in edge filter");
        mask |= kind_bit(*kind);
    }
    return mask;
}

}  // namespace

ThreadPriority parse_priority(std::string_view name) {
    if (name == "low") return ThreadPriority::Low;
    if (name == "normal" || name.empty()) return ThreadPriority::Normal;
    if (name == "high") return ThreadPriority::High;
    if (name == "realtime") return ThreadPriority::Realtime;
    throw ConfigError("unknown thread priority '" + std::string(name) + "'");
}

const StageSpec* PipelineSpec::find_stage(std::string_view stage_name) const {
    for (const StageSpec& s : stages) {
        if (s.name == stage_name) return &s;
    }
    return nullptr;
}

StageSpec* PipelineSpec::find_stage(std::string_view stage_name) {
    for (StageSpec& s : stages) {
        if (s.name == stage_name) return &s;
    }
    return nullptr;
}

void PipelineSpec::apply_device_default() {
    if (device.empty()) return;
    for (StageSpec& s : stages) {
        if (!s.params.has("device")) s.params.set("device", device);
    }
}

PipelineSpec parse_pipeline(std::string_view yaml_text, const std::filesystem::path& config_dir) {
    YAML::Node root;
    try {
        root = YAML::Load(std::string(yaml_text));
    } catch (const YAML::Exception& e) {
        throw ConfigError(std::string("pipeline YAML: ") + e.what());
    }
    if (!root.IsMap()) throw ConfigError("pipeline YAML must be a mapping");

    const std::string dir = config_dir.empty() ? std::string(".") : config_dir.generic_string();
    PipelineSpec spec;

    if (const auto p = root["pipeline"]) {
        if (p["name"]) spec.name = scalar(p["name"], "pipeline.name", dir);
        if (p["source_language"]) spec.source_language = scalar(p["source_language"], "source_language", dir);
        if (p["target_language"]) spec.target_language = scalar(p["target_language"], "target_language", dir);
        if (p["sample_rate"]) spec.sample_rate = scalar_as<int>(p["sample_rate"], "sample_rate");
        if (p["models"]) spec.models_manifest = scalar(p["models"], "pipeline.models", dir);
        if (p["device"]) spec.device = scalar(p["device"], "pipeline.device", dir);
    }

    if (const auto t = root["telemetry"]) {
        if (t["flush_ms"]) spec.telemetry.flush_ms = scalar_as<int>(t["flush_ms"], "telemetry.flush_ms");
        if (t["prometheus"]) spec.telemetry.prometheus_path = scalar(t["prometheus"], "telemetry.prometheus", dir);
        if (t["trace"]) spec.telemetry.trace_path = scalar(t["trace"], "telemetry.trace", dir);
    }

    if (const auto threads = root["threads"]) {
        for (const auto& t : threads) {
            ThreadSpec ts;
            ts.name = scalar(t["name"], "threads[].name", dir);
            if (t["core"] && t["cores"]) throw ConfigError("thread '" + ts.name + "': give 'core' or 'cores', not both");
            if (t["core"]) ts.cores.push_back(scalar_as<int>(t["core"], "threads[].core"));
            if (const auto cores = t["cores"]) {
                if (!cores.IsSequence()) throw ConfigError("thread '" + ts.name + "': 'cores' must be a list");
                for (const auto& c : cores) ts.cores.push_back(scalar_as<int>(c, "threads[].cores[]"));
            }
            for (int c : ts.cores) {
                if (c < 0) throw ConfigError("thread '" + ts.name + "': negative core " + std::to_string(c));
            }
            if (t["priority"]) ts.priority = parse_priority(scalar(t["priority"], "priority", dir));
            spec.threads.push_back(std::move(ts));
        }
    }

    const auto stages = root["stages"];
    if (!stages || !stages.IsSequence()) throw ConfigError("pipeline YAML needs a 'stages' list");
    for (const auto& s : stages) {
        StageSpec ss;
        ss.name = scalar(s["name"], "stages[].name", dir);
        ss.type = s["type"] ? scalar(s["type"], "stages[].type", dir) : ss.name;
        if (s["thread"]) ss.thread = scalar(s["thread"], "stages[].thread", dir);
        if (s["tick_ms"]) ss.tick_ms = scalar_as<int>(s["tick_ms"], "stages[].tick_ms");
        if (const auto params = s["params"]) flatten(params, "", ss.params, dir);
        spec.stages.push_back(std::move(ss));
    }

    // Without explicit threads everything shares one thread (fine for offline runs).
    if (spec.threads.empty()) spec.threads.push_back({"main", {}, ThreadPriority::Normal});
    for (StageSpec& s : spec.stages) {
        if (s.thread.empty()) s.thread = spec.threads.front().name;
    }

    if (const auto edges = root["edges"]) {
        for (const auto& e : edges) {
            const std::string from = scalar(e["from"], "edges[].from", dir);
            const std::uint32_t kinds = parse_kinds(e["kinds"]);
            const std::size_t capacity =
                e["capacity"] ? scalar_as<std::size_t>(e["capacity"], "edges[].capacity") : 64;
            const bool feedback = e["feedback"] && scalar_as<bool>(e["feedback"], "edges[].feedback");
            const auto targets = string_list(e["to"], dir);
            if (targets.empty()) throw ConfigError("edge from '" + from + "' has no 'to'");
            for (const std::string& to : targets) {
                spec.edges.push_back({from, to, kinds, capacity, feedback});
            }
        }
    }

    validate(spec);
    return spec;
}

PipelineSpec load_pipeline(const std::filesystem::path& yaml_path) {
    std::ifstream in(yaml_path, std::ios::binary);
    if (!in) throw ConfigError("cannot open pipeline config '" + yaml_path.string() + "'");
    std::ostringstream text;
    text << in.rdbuf();
    auto dir = yaml_path.parent_path();
    if (dir.empty()) dir = ".";
    return parse_pipeline(text.str(), dir);
}

void apply_override(PipelineSpec& spec, std::string_view key, std::string_view value) {
    const auto dot = key.find('.');
    if (dot == std::string_view::npos || dot == 0 || dot + 1 == key.size()) {
        throw ConfigError("override key must look like <stage>.<param>: '" + std::string(key) + "'");
    }
    const std::string_view head = key.substr(0, dot);
    const std::string_view rest = key.substr(dot + 1);
    const std::string v(value);

    if (head == "pipeline") {
        if (rest == "name") spec.name = v;
        else if (rest == "source_language") spec.source_language = v;
        else if (rest == "target_language") spec.target_language = v;
        else if (rest == "sample_rate") spec.sample_rate = std::stoi(v);
        else if (rest == "models") spec.models_manifest = v;
        else if (rest == "device") spec.device = v;
        else throw ConfigError("unknown pipeline setting '" + std::string(rest) + "'");
        return;
    }
    if (head == "telemetry") {
        if (rest == "flush_ms") spec.telemetry.flush_ms = std::stoi(v);
        else if (rest == "prometheus") spec.telemetry.prometheus_path = v;
        else if (rest == "trace") spec.telemetry.trace_path = v;
        else throw ConfigError("unknown telemetry setting '" + std::string(rest) + "'");
        return;
    }
    StageSpec* stage = spec.find_stage(head);
    if (stage == nullptr) throw ConfigError("override for unknown stage '" + std::string(head) + "'");
    if (rest == "tick_ms") stage->tick_ms = std::stoi(v);
    else if (rest == "thread") stage->thread = v;
    else stage->params.set(std::string(rest), v);
}

void validate(const PipelineSpec& spec) {
    if (spec.stages.empty()) throw ConfigError("pipeline has no stages");
    if (spec.sample_rate <= 0) throw ConfigError("pipeline.sample_rate must be positive");

    std::map<std::string, const ThreadSpec*, std::less<>> threads;
    for (const ThreadSpec& t : spec.threads) {
        if (!threads.emplace(t.name, &t).second) throw ConfigError("duplicate thread '" + t.name + "'");
    }
    std::set<std::string, std::less<>> stage_names;
    for (const StageSpec& s : spec.stages) {
        if (s.name.empty()) throw ConfigError("stage with empty name");
        if (!stage_names.insert(s.name).second) throw ConfigError("duplicate stage '" + s.name + "'");
        const auto thread = threads.find(s.thread);
        if (thread == threads.end()) {
            throw ConfigError("stage '" + s.name + "' uses unknown thread '" + s.thread + "'");
        }
        if (s.tick_ms < 0) throw ConfigError("stage '" + s.name + "' has negative tick_ms");
        // Workers an engine starts from a pinned thread inherit its cores (whisper.cpp starts them
        // on every decode). Four spinning workers on one core are far slower than one worker.
        const std::size_t cores = thread->second->cores.size();
        const std::int64_t workers = s.params.integer("threads", 0);
        if (cores > 0 && workers > static_cast<std::int64_t>(cores)) {
            throw ConfigError("stage '" + s.name + "' runs " + std::to_string(workers) + " worker threads but thread '" +
                              s.thread + "' is pinned to " + std::to_string(cores) +
                              " core(s), which the workers inherit: list more cores or lower 'threads'");
        }
    }
    for (const EdgeSpec& e : spec.edges) {
        if (!stage_names.contains(e.from)) throw ConfigError("edge from unknown stage '" + e.from + "'");
        if (!stage_names.contains(e.to)) throw ConfigError("edge to unknown stage '" + e.to + "'");
        if (e.from == e.to) throw ConfigError("self-edge on stage '" + e.from + "'");
        if (e.capacity < 2) throw ConfigError("edge capacity must be at least 2");
    }
    (void)topological_order(spec);  // throws on cycles
}

std::vector<std::size_t> topological_order(const PipelineSpec& spec) {
    const std::size_t n = spec.stages.size();
    std::unordered_map<std::string, std::size_t> index;
    for (std::size_t i = 0; i < n; ++i) index[spec.stages[i].name] = i;

    std::vector<std::vector<std::size_t>> out(n);
    std::vector<std::size_t> indegree(n, 0);
    for (const EdgeSpec& e : spec.edges) {
        if (e.feedback) continue;
        const auto from = index.find(e.from);
        const auto to = index.find(e.to);
        if (from == index.end() || to == index.end()) continue;
        out[from->second].push_back(to->second);
        ++indegree[to->second];
    }

    // Kahn's algorithm; the min-heap keeps declaration order among ready stages (deterministic).
    std::priority_queue<std::size_t, std::vector<std::size_t>, std::greater<>> ready;
    for (std::size_t i = 0; i < n; ++i) {
        if (indegree[i] == 0) ready.push(i);
    }
    std::vector<std::size_t> order;
    order.reserve(n);
    while (!ready.empty()) {
        const std::size_t i = ready.top();
        ready.pop();
        order.push_back(i);
        for (std::size_t j : out[i]) {
            if (--indegree[j] == 0) ready.push(j);
        }
    }
    if (order.size() != n) {
        throw ConfigError("pipeline graph has a cycle; mark the loop-closing edge 'feedback: true'");
    }
    return order;
}

}  // namespace ee
