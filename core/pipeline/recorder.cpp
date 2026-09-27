#include "core/pipeline/recorder.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <numeric>

#include "core/runtime/log.hpp"
#include "core/telemetry/telemetry.hpp"
#include "core/translate/languages.hpp"

namespace ee {

using json = nlohmann::json;

float UtteranceRecord::ecs_mean() const {
    if (ecs.empty()) return 0.0f;
    return std::accumulate(ecs.begin(), ecs.end(), 0.0f) / static_cast<float>(ecs.size());
}

void RecorderStage::open(StageContext& ctx) {
    ctx_ = &ctx;
    srt_path_ = ctx.params().str("srt");
    json_path_ = ctx.params().str("json");
    emotion_tags_ = ctx.params().flag("emotion_tags", true);
}

UtteranceRecord& RecorderStage::record(std::uint64_t id) {
    UtteranceRecord& r = records_[id];
    r.id = id;
    return r;
}

void RecorderStage::process(Frame& f) {
    if (f.utterance == 0) return;
    switch (f.kind) {
    case FrameKind::Transcript:
        if (f.is_final()) {
            UtteranceRecord& r = record(f.utterance);
            r.source_text = f.text;
            r.source_language = f.language;
            r.words = f.words;
            r.src_start = f.src_start;
            r.src_end = f.src_end;
        }
        break;
    case FrameKind::Utterance: {
        UtteranceRecord& r = record(f.utterance);
        r.source_emphasis = f.emphasis;
        r.emotion = f.emotion;
        r.axis_confidence = f.axis_confidence;
        break;
    }
    case FrameKind::Translation:
        if (f.is_final()) {
            UtteranceRecord& r = record(f.utterance);
            r.mt_input = f.detail;
            r.translation = f.text;
            r.target_language = f.language;
            r.target_emphasis = f.emphasis;
            r.emotion = f.emotion;
        }
        break;
    case FrameKind::Speech: {
        UtteranceRecord& r = record(f.utterance);
        r.prosody = f.prosody;
        r.correction = f.delta;
        break;
    }
    case FrameKind::Feedback: {
        UtteranceRecord& r = record(f.utterance);
        r.ecs.push_back(f.score);
        r.output_emotion.push_back(f.emotion);
        break;
    }
    case FrameKind::Playout: {
        UtteranceRecord& r = record(f.utterance);
        r.out_start = f.out_start;
        r.out_end = f.out_end;
        break;
    }
    default: break;
    }
}

std::vector<UtteranceRecord> RecorderStage::records() const {
    std::vector<UtteranceRecord> out;
    out.reserve(records_.size());
    for (const auto& [id, r] : records_) out.push_back(r);
    return out;
}

void RecorderStage::close() {
    const auto recs = records();
    const auto write = [](const std::string& path, const std::string& text, const char* what) {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (!out) {
            log::warn("recorder: cannot write ", what, " to '", path, "'");
            return;
        }
        out << text;
    };
    if (!srt_path_.empty()) write(srt_path_, to_srt(recs, emotion_tags_), "captions");
    if (!json_path_.empty()) {
        write(json_path_, to_session_json(recs, ctx_->pipeline(), ctx_->services().telemetry), "session");
    }
}

namespace {

std::string srt_time(double seconds) {
    const auto ms_total = static_cast<long long>(std::llround(std::max(0.0, seconds) * 1000.0));
    const long long h = ms_total / 3'600'000;
    const long long m = ms_total / 60'000 % 60;
    const long long s = ms_total / 1000 % 60;
    const long long ms = ms_total % 1000;
    char buf[32];
    std::snprintf(buf, sizeof buf, "%02lld:%02lld:%02lld,%03lld", h, m, s, ms);
    return buf;
}

json vad_json(const Vad& v) { return {{"valence", v.v}, {"arousal", v.a}, {"dominance", v.d}}; }

json emphasized_words(const std::vector<std::string>& words, const std::vector<std::uint16_t>& indices) {
    json out = json::array();
    for (std::uint16_t i : indices) {
        if (i < words.size()) out.push_back(words[i]);
    }
    return out;
}

json snapshot_ms(const telemetry::Histogram::Snapshot& s) {
    return {{"count", s.count},
            {"p50_ms", static_cast<double>(s.percentile(0.5)) / 1000.0},
            {"p95_ms", static_cast<double>(s.percentile(0.95)) / 1000.0}};
}

}  // namespace

std::string to_srt(const std::vector<UtteranceRecord>& records, bool emotion_tags) {
    std::string out;
    int index = 1;
    for (const UtteranceRecord& r : records) {
        if (r.translation.empty()) continue;
        const bool played = r.out_start >= 0.0 && r.out_end > r.out_start;
        const double start = played ? r.out_start : r.src_start;
        const double end = played ? r.out_end : std::max(r.src_end, r.src_start + 1.0);
        out += std::to_string(index++) + "\n";
        out += srt_time(start) + " --> " + srt_time(end) + "\n";
        if (emotion_tags) out += "[" + std::string(to_string(r.emotion.label)) + "] ";
        out += r.translation + "\n\n";
    }
    return out;
}

std::string to_session_json(const std::vector<UtteranceRecord>& records, const PipelineSpec& spec,
                            const telemetry::Telemetry* telemetry) {
    json root;
    root["pipeline"] = spec.name;
    root["source_language"] = spec.source_language;
    root["target_language"] = spec.target_language;
    json utterances = json::array();
    for (const UtteranceRecord& r : records) {
        std::vector<std::string> source_words;
        json words = json::array();
        for (const Word& w : r.words) {
            source_words.push_back(w.text);
            words.push_back({{"text", w.text}, {"start", r.src_start + w.t0}, {"end", r.src_start + w.t1}});
        }
        const std::vector<std::string> target_words = split_words(r.translation, r.target_language);
        json u;
        u["id"] = r.id;
        u["source"] = {{"text", r.source_text}, {"language", r.source_language}, {"start", r.src_start},
                       {"end", r.src_end}, {"words", words}, {"emphasis", emphasized_words(source_words, r.source_emphasis)}};
        u["emotion"] = vad_json(r.emotion.vad);
        u["emotion"]["label"] = to_string(r.emotion.label);
        u["emotion"]["confidence"] = r.emotion.confidence;
        u["mt_input"] = r.mt_input;
        u["translation"] = {{"text", r.translation}, {"language", r.target_language},
                            {"emphasis", emphasized_words(target_words, r.target_emphasis)}};
        u["prosody"] = {{"pitch_pct", r.prosody.pitch_pct}, {"range_pct", r.prosody.range_pct},
                        {"rate_pct", r.prosody.rate_pct},   {"energy_db", r.prosody.energy_db},
                        {"pause_ms", r.prosody.pause_ms},   {"accent", r.prosody.accent},
                        {"final_fall", r.prosody.final_fall}, {"hesitation", r.prosody.hesitation},
                        {"tension", r.prosody.tension}};
        u["closed_loop_correction"] = vad_json(r.correction);
        u["ecs"] = r.ecs;
        if (!r.ecs.empty()) u["ecs_mean"] = r.ecs_mean();
        json heard = json::array();  // what 5.2 measured on the synthesized clauses
        for (const EmotionState& e : r.output_emotion) {
            json one = vad_json(e.vad);
            one["label"] = to_string(e.label);
            heard.push_back(std::move(one));
        }
        u["output_emotion"] = std::move(heard);
        if (r.out_start >= 0.0) u["output"] = {{"start", r.out_start}, {"end", r.out_end}};
        utterances.push_back(std::move(u));
    }
    root["utterances"] = std::move(utterances);

    if (telemetry != nullptr) {
        json latency = json::array();
        for (const auto& row : telemetry->budget_report()) {
            json j = snapshot_ms(row.snapshot);
            j["row"] = row.row->name;
            j["target_ms"] = static_cast<double>(row.row->target_us) / 1000.0;
            latency.push_back(std::move(j));
        }
        root["latency_budget"] = std::move(latency);
        root["end_to_end"] = snapshot_ms(telemetry->end_to_end());
        const auto ecs = telemetry->ecs();
        root["ecs_summary"] = {{"count", ecs.count}, {"mean", ecs.mean() / 10000.0},
                               {"p50", static_cast<double>(ecs.percentile(0.5)) / 10000.0}};
        const auto rtf = telemetry->rtf();
        root["asr_rtf"] = {{"count", rtf.count}, {"p50", static_cast<double>(rtf.percentile(0.5)) / 10000.0},
                           {"p95", static_cast<double>(rtf.percentile(0.95)) / 10000.0}};
    }
    return root.dump(2);
}

}  // namespace ee
