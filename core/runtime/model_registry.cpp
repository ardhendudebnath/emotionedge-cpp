#include "core/runtime/model_registry.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <set>
#include <sstream>
#include <system_error>

#include "core/runtime/log.hpp"
#include "core/runtime/sha256.hpp"

namespace ee {

namespace fs = std::filesystem;
using json = nlohmann::json;

std::string_view to_string(VerifyStatus status) noexcept {
    switch (status) {
    case VerifyStatus::Ok: return "ok";
    case VerifyStatus::Missing: return "missing";
    case VerifyStatus::SizeMismatch: return "size mismatch";
    case VerifyStatus::HashMismatch: return "sha256 mismatch";
    case VerifyStatus::Unpinned: return "unpinned";
    }
    return "?";
}

namespace {

std::string get_string(const json& j, const char* key) {
    const auto it = j.find(key);
    return it != j.end() && it->is_string() ? it->get<std::string>() : std::string();
}

std::vector<std::string> get_strings(const json& j, const char* key) {
    std::vector<std::string> out;
    const auto it = j.find(key);
    if (it == j.end()) return out;
    if (it->is_string()) {
        out.push_back(it->get<std::string>());
    } else if (it->is_array()) {
        for (const auto& v : *it) {
            if (v.is_string()) out.push_back(v.get<std::string>());
        }
    }
    return out;
}

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

bool covers(const std::vector<std::string>& languages, std::string_view lang) {
    if (languages.empty()) return true;
    return std::any_of(languages.begin(), languages.end(),
                       [&](const std::string& l) { return l == "*" || l == lang; });
}

std::int64_t mtime_ticks(const fs::path& p) {
    std::error_code ec;
    const auto t = fs::last_write_time(p, ec);
    return ec ? 0 : static_cast<std::int64_t>(t.time_since_epoch().count());
}

}  // namespace

ModelRegistry ModelRegistry::parse(std::string_view text, const fs::path& base_dir) {
    json root;
    try {
        root = json::parse(text);
    } catch (const json::exception& e) {
        throw ConfigError(std::string("model manifest: ") + e.what());
    }
    if (!root.is_object()) throw ConfigError("model manifest must be a JSON object");

    ModelRegistry reg;
    std::set<std::string> ids;
    if (const auto it = root.find("models"); it != root.end()) {
        if (!it->is_array()) throw ConfigError("manifest 'models' must be an array");
        for (const json& m : *it) {
            ModelEntry e;
            e.id = get_string(m, "id");
            e.task = get_string(m, "task");
            e.format = get_string(m, "format");
            e.path = get_string(m, "path");
            e.check_file = get_string(m, "check_file");
            e.sha256 = lower(get_string(m, "sha256"));
            if (const auto b = m.find("bytes"); b != m.end() && b->is_number_unsigned()) {
                e.bytes = b->get<std::uint64_t>();
            }
            e.languages = get_strings(m, "languages");
            e.devices = get_strings(m, "devices");
            e.precision = get_string(m, "precision");
            e.license = get_string(m, "license");
            e.source = get_string(m, "source");
            if (e.id.empty() || e.path.empty()) throw ConfigError("manifest model entries need 'id' and 'path'");
            if (!ids.insert(e.id).second) throw ConfigError("duplicate model id '" + e.id + "'");
            const fs::path p(e.path);
            e.resolved = p.is_absolute() ? p : (base_dir / p).lexically_normal();
            reg.models_.push_back(std::move(e));
        }
    }
    if (const auto it = root.find("device_profiles"); it != root.end() && it->is_object()) {
        for (const auto& [name, d] : it->items()) {
            DeviceProfile profile;
            profile.name = name;
            profile.execution_providers = get_strings(d, "execution_providers");
            if (const auto t = d.find("threads"); t != d.end() && t->is_number_integer()) {
                profile.threads = t->get<int>();
            }
            profile.precision = get_string(d, "precision");
            reg.devices_.push_back(std::move(profile));
        }
    }
    if (const auto it = root.find("language_packs"); it != root.end() && it->is_object()) {
        for (const auto& [pair, p] : it->items()) {
            const auto dash = pair.find('-');
            if (dash == std::string::npos) throw ConfigError("language pack '" + pair + "' must be named <src>-<tgt>");
            LanguagePack pack;
            pack.source = pair.substr(0, dash);
            pack.target = pair.substr(dash + 1);
            for (const auto& [task, id] : p.items()) {
                if (!id.is_string()) continue;
                const std::string model_id = id.get<std::string>();
                if (!ids.contains(model_id)) {
                    throw ConfigError("language pack '" + pair + "' references unknown model '" + model_id + "'");
                }
                pack.models[task] = model_id;
            }
            reg.packs_.push_back(std::move(pack));
        }
    }
    return reg;
}

ModelRegistry ModelRegistry::load(const fs::path& manifest) {
    std::ifstream in(manifest, std::ios::binary);
    if (!in) throw ConfigError("cannot open model manifest '" + manifest.string() + "'");
    std::ostringstream text;
    text << in.rdbuf();
    return parse(text.str(), manifest.parent_path());
}

const ModelEntry* ModelRegistry::find(std::string_view id) const {
    for (const ModelEntry& e : models_) {
        if (e.id == id) return &e;
    }
    return nullptr;
}

const LanguagePack* ModelRegistry::language_pack(std::string_view source, std::string_view target) const {
    for (const LanguagePack& p : packs_) {
        if (p.source == source && p.target == target) return &p;
    }
    return nullptr;
}

const ModelEntry* ModelRegistry::resolve(std::string_view task, std::string_view source,
                                         std::string_view target) const {
    if (const LanguagePack* pack = language_pack(source, target)) {
        if (const auto it = pack->models.find(task); it != pack->models.end()) {
            if (const ModelEntry* e = find(it->second)) return e;
        }
    }
    for (const ModelEntry& m : models_) {
        if (m.task != task) continue;
        if (task == "mt") {
            if (covers(m.languages, source) && covers(m.languages, target)) return &m;
        } else if (task == "tts") {
            if (covers(m.languages, target)) return &m;
        } else if (covers(m.languages, source)) {
            return &m;
        }
    }
    return nullptr;
}

const DeviceProfile* ModelRegistry::device_profile(std::string_view name) const {
    for (const DeviceProfile& d : devices_) {
        if (d.name == name) return &d;
    }
    return nullptr;
}

VerifyResult ModelRegistry::verify(const ModelEntry& entry, bool use_stamp) const {
    VerifyResult r;
    r.entry = &entry;
    const fs::path file = entry.check_file.empty() ? entry.resolved : entry.resolved / entry.check_file;
    std::error_code ec;
    if (!fs::is_regular_file(file, ec)) {
        r.status = VerifyStatus::Missing;
        return r;
    }
    const std::uint64_t size = fs::file_size(file, ec);
    if (ec || (entry.bytes != 0 && size != entry.bytes)) {
        r.status = VerifyStatus::SizeMismatch;
        return r;
    }

    // The stamp is a cache inside the same trust boundary as the model directory.
    const fs::path stamp = file.string() + ".sha256";
    const std::int64_t mtime = mtime_ticks(file);
    if (use_stamp) {
        std::ifstream in(stamp);
        std::string hash;
        std::uint64_t stamp_size = 0;
        std::int64_t stamp_mtime = 0;
        if (in >> hash >> stamp_size >> stamp_mtime && stamp_size == size && stamp_mtime == mtime) {
            r.actual_sha256 = hash;
        }
    }
    if (r.actual_sha256.empty()) {
        const auto hash = sha256_file(file);
        if (!hash) {
            r.status = VerifyStatus::Missing;
            return r;
        }
        r.actual_sha256 = *hash;
        std::ofstream out(stamp, std::ios::trunc);
        if (out) out << r.actual_sha256 << ' ' << size << ' ' << mtime << '\n';
    }
    if (entry.sha256.empty()) {
        r.status = VerifyStatus::Unpinned;
    } else {
        r.status = r.actual_sha256 == entry.sha256 ? VerifyStatus::Ok : VerifyStatus::HashMismatch;
    }
    return r;
}

std::vector<VerifyResult> ModelRegistry::verify_all(bool use_stamp) const {
    std::vector<VerifyResult> out;
    out.reserve(models_.size());
    for (const ModelEntry& e : models_) out.push_back(verify(e, use_stamp));
    return out;
}

std::string resolve_model_path(const Params& params, const ModelRegistry* registry, std::string_view key) {
    const std::string id_key = std::string(key) + "_id";
    const std::string id = params.str(id_key);
    if (id.empty()) return params.str(key);
    if (registry == nullptr) {
        throw ConfigError("'" + id_key + ": " + id + "' needs a model manifest (pipeline.models)");
    }
    const ModelEntry* entry = registry->find(id);
    if (entry == nullptr) throw ConfigError("model id '" + id + "' is not in the manifest");
    const VerifyResult r = registry->verify(*entry);
    switch (r.status) {
    case VerifyStatus::Ok: break;
    case VerifyStatus::Unpinned:
        log::warn("model '", id, "' has no pinned sha256 in the manifest (actual ", r.actual_sha256, ")");
        break;
    case VerifyStatus::Missing:
        throw ConfigError("model '" + id + "' not found at " + entry->resolved.string() +
                          " (see models/README.md to fetch it)");
    default:
        throw ConfigError("model '" + id + "' failed verification: " + std::string(to_string(r.status)));
    }
    return entry->resolved.string();
}

}  // namespace ee
