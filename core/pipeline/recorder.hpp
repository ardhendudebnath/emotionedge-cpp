#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "core/runtime/stage.hpp"

namespace ee {

/// Everything the pipeline decided about one utterance, for captions and the session export.
struct UtteranceRecord {
    std::uint64_t id = 0;
    double src_start = 0.0;
    double src_end = 0.0;
    std::string source_text;
    std::string source_language;
    std::vector<Word> words;
    std::vector<std::uint16_t> source_emphasis;
    EmotionState emotion;
    Vad axis_confidence;
    std::string mt_input;
    std::string translation;
    std::string target_language;
    std::vector<std::uint16_t> target_emphasis;
    ProsodyTargets prosody;
    Vad correction;               ///< closed-loop correction in effect for this utterance
    std::vector<float> ecs;       ///< one score per synthesized clause
    double out_start = -1.0;      ///< output timeline (seconds), -1 if never played
    double out_end = -1.0;

    [[nodiscard]] float ecs_mean() const;
};

/// Stage 5.3 "Outputs & Interfaces": collects transcripts, emotion, translations, prosody plans,
/// ECS scores and playout times per utterance, and at the end writes live-caption files:
/// SRT captions with emotion tags and a JSON session export (the WAV comes from playback).
class RecorderStage final : public IStage {
public:
    void open(StageContext& ctx) override;
    void process(Frame& frame) override;
    void close() override;

    [[nodiscard]] std::vector<UtteranceRecord> records() const;

private:
    UtteranceRecord& record(std::uint64_t id);

    StageContext* ctx_ = nullptr;
    std::map<std::uint64_t, UtteranceRecord> records_;
    std::string srt_path_;
    std::string json_path_;
    bool emotion_tags_ = true;
};

/// SRT text for the records (target-language captions, `[anger] ...` tags when enabled).
[[nodiscard]] std::string to_srt(const std::vector<UtteranceRecord>& records, bool emotion_tags = true);

/// JSON session export.
[[nodiscard]] std::string to_session_json(const std::vector<UtteranceRecord>& records, const PipelineSpec& spec,
                                          const telemetry::Telemetry* telemetry);

}  // namespace ee
