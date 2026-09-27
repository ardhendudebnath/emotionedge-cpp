#include "core/asr/scripted_engine.hpp"

#include <nlohmann/json.hpp>

#include <fstream>
#include <sstream>

#include "core/runtime/params.hpp"

namespace ee {

using json = nlohmann::json;

ScriptedAsrEngine::ScriptedAsrEngine(std::vector<ScriptWord> words, std::string language)
    : words_(std::move(words)), language_(std::move(language)) {}

ScriptedAsrEngine ScriptedAsrEngine::from_json(std::string_view text) {
    json root;
    try {
        root = json::parse(text);
    } catch (const json::exception& e) {
        throw ConfigError(std::string("ASR script: ") + e.what());
    }
    std::vector<ScriptWord> words;
    for (const json& w : root.value("words", json::array())) {
        ScriptWord sw;
        sw.text = w.value("w", std::string());
        sw.t0 = w.value("t0", 0.0);
        sw.t1 = w.value("t1", sw.t0);
        if (sw.text.empty() || sw.t1 < sw.t0) throw ConfigError("ASR script words need 'w' and t0 <= t1");
        words.push_back(std::move(sw));
    }
    return ScriptedAsrEngine(std::move(words), root.value("language", std::string("en")));
}

ScriptedAsrEngine ScriptedAsrEngine::from_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw ConfigError("cannot open ASR script '" + path.string() + "'");
    std::ostringstream text;
    text << in.rdbuf();
    return from_json(text.str());
}

AsrResult ScriptedAsrEngine::transcribe(const AsrRequest& r) {
    AsrResult out;
    out.language = language_;
    const double start = r.utterance_start_s;
    const double end = start + static_cast<double>(r.audio.size()) / r.sample_rate;
    for (const ScriptWord& w : words_) {
        const double mid = 0.5 * (w.t0 + w.t1);
        if (mid < start || mid >= end) continue;
        if (!r.final && w.t1 > end) continue;  // a partial only has words that were fully heard
        out.words.push_back({w.text, static_cast<float>(w.t0 - start), static_cast<float>(w.t1 - start), 1.0f});
    }
    return out;
}

std::string make_asr_script(const std::vector<Word>& words, double offset_s, std::string_view language) {
    json root;
    root["language"] = std::string(language);
    json list = json::array();
    for (const Word& w : words) {
        list.push_back({{"w", w.text}, {"t0", offset_s + w.t0}, {"t1", offset_s + w.t1}});
    }
    root["words"] = std::move(list);
    return root.dump(2);
}

}  // namespace ee
