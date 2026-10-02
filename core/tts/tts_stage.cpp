#include "core/tts/tts_stage.hpp"

#include <algorithm>

#include "core/audio/audio_io.hpp"
#include "core/runtime/log.hpp"
#include "core/telemetry/telemetry.hpp"
#include "core/tts/formant_synth.hpp"

namespace ee {

std::unique_ptr<ITtsEngine> make_tts_engine(const Params& params, const ModelRegistry* registry) {
    const std::string kind = params.str("engine", "formant");
    if (kind == "formant") {
        FormantVoice voice;
        voice.sample_rate = static_cast<int>(params.integer("sample_rate", voice.sample_rate));
        voice.base_f0_hz = params.real("base_f0_hz", voice.base_f0_hz);
        voice.level_dbfs = params.real("level_dbfs", voice.level_dbfs);
        return std::make_unique<FormantSynth>(voice);
    }
    if (kind == "piper") {
#if defined(EE_HAVE_PIPER)
        const std::string model = resolve_model_path(params, registry, "model");
        if (model.empty()) throw ConfigError("tts engine 'piper' needs a 'model' (.onnx voice) or 'model_id'");
        return make_piper_engine(model, params, registry);
#else
        (void)registry;
        throw ConfigError("tts engine 'piper' needs a build with -DEE_WITH_PIPER=ON");
#endif
    }
    if (kind == "kokoro") {
#if defined(EE_HAVE_PIPER)
        const std::string dir = resolve_model_path(params, registry, "model");
        if (dir.empty()) throw ConfigError("tts engine 'kokoro' needs a 'model' (voice directory) or 'model_id'");
        return make_kokoro_engine(dir, params, registry);
#else
        (void)registry;
        throw ConfigError("tts engine 'kokoro' needs a build with -DEE_WITH_PIPER=ON (ONNX Runtime + espeak-ng)");
#endif
    }
    throw ConfigError("unknown tts engine '" + kind + "' (formant | piper | kokoro)");
}

namespace {

/// UTF-8 code points: a length measure that treats Devanagari (3 bytes) and Latin (1) alike.
std::size_t characters(std::string_view text) {
    return static_cast<std::size_t>(
        std::count_if(text.begin(), text.end(), [](char c) { return (static_cast<unsigned char>(c) & 0xC0) != 0x80; }));
}

/// The speed an engine renders at for these prosody targets (every engine reads rate_pct so).
double engine_speed(const ProsodyTargets& p, float pace) {
    return std::clamp((1.0 + p.rate_pct / 100.0) * pace, 0.5, 2.0);
}

}  // namespace

float PacingConfig::speed(double backlog_s, double predicted_s, double source_s, double gap_s) const noexcept {
    if (max_speed <= 1.0f || full_s <= start_s) return 1.0f;
    const double t = std::clamp((backlog_s - start_s) / (full_s - start_s), 0.0, 1.0);
    double s = 1.0 + t * (max_speed - 1.0);
    if (fit_next && predicted_s > 0.0 && source_s + gap_s > 0.0) s = std::max(s, predicted_s / (source_s + gap_s));
    return static_cast<float>(std::min(s, static_cast<double>(max_speed)));
}

void TtsStage::open(StageContext& ctx) {
    ctx_ = &ctx;
    const Params& p = ctx.params();
    engine_.publish(std::shared_ptr<ITtsEngine>(make_tts_engine(p, ctx.services().models)));
    const auto engine = engine_.acquire();
    chunker_.min_first_chars = static_cast<std::size_t>(p.integer("min_first_chars", 8));
    chunker_.min_chars = static_cast<std::size_t>(p.integer("min_chars", 24));
    chunker_.max_chars = static_cast<std::size_t>(p.integer("max_chars", 140));
    chunker_.max_first_words = static_cast<std::size_t>(p.integer("max_first_words", 0));
    chunk_samples_ = static_cast<std::size_t>(engine->sample_rate() * p.integer("chunk_ms", 100) / 1000);
    use_voice_print_ = p.flag("use_voice_print", true);
    pacing_.start_s = p.real("pacing.start_ms", 250.0f) / 1000.0f;
    pacing_.full_s = p.real("pacing.full_ms", 1500.0f) / 1000.0f;
    pacing_.max_speed = p.real("pacing.max_speed", 1.0f);
    pacing_.fit_next = p.flag("pacing.fit_next", false);
    gap_s_ = p.real("pacing.default_gap_ms", 1000.0f) / 1000.0f;  // until the speaker's own gaps are seen
    if (pacing_.max_speed < 1.0f || pacing_.max_speed > 2.0f || pacing_.full_s <= pacing_.start_s || gap_s_ < 0.0) {
        throw ConfigError("tts pacing needs 1 <= max_speed <= 2, full_ms > start_ms and default_gap_ms >= 0");
    }
    pace_metric_ = &ctx.services().telemetry->metrics().histogram(
        "ee_tts_pacing_speed", "Speaking-rate multiplier from adaptive pacing, per utterance", {}, 1e-4);

    if (p.flag("calibration", true)) {
        // Neutral reference render of this voice, for 5.2's baseline. Playback ignores it.
        const std::string text = p.str("calibration_text", "The quick brown fox jumps over the lazy dog, and then it rests.");
        SynthesisRequest req;
        req.text = text;
        req.language = ctx.pipeline().target_language;
        engine->synthesize(req, result_);
        // A first estimate of this voice's pace for adaptive pacing, until real clauses refine it.
        if (result_.sample_rate > 0) {
            seconds_per_char_ = static_cast<double>(result_.audio.size()) / result_.sample_rate /
                                static_cast<double>(std::max<std::size_t>(1, characters(text)));
        }
        Frame& f = ctx.make(FrameKind::SynthAudio);
        f.flags = frame_flags::kCalibration;
        f.sample_rate = result_.sample_rate;
        f.language = ctx.pipeline().target_language;
        f.text = text;
        f.audio = result_.audio;
        ctx.emit(f);
    }
}

void TtsStage::process(Frame& f) {
    switch (f.kind) {
    case FrameKind::VoicePrint:
        voices_[f.utterance] = {f.voice, f.voice_f0};
        latest_voice_ = {f.voice, f.voice_f0};
        have_voice_ = true;
        while (voices_.size() > 8) voices_.erase(voices_.begin());
        return;
    case FrameKind::Control:
        if (f.has(frame_flags::kBargeIn)) {
            std::erase_if(pending_, [&](const Job& j) { return j.speech.utterance < f.utterance; });
        }
        return;
    case FrameKind::Speech: {
        const std::vector<Clause> clauses = chunk_clauses(f.text, f.language, f.emphasis, chunker_);
        // Decided before this utterance's own audio enters the queue: later in the utterance,
        // its earlier clauses would read as backlog.
        float pace = 1.0f;
        if (pacing_.max_speed > 1.0f) {
            // The gap before this utterance updates the expectation for the one after it
            // (clamped: one long pause should not switch pacing off for good).
            if (last_src_end_ >= 0.0 && f.src_start >= last_src_end_) {
                gap_s_ = 0.5 * gap_s_ + 0.5 * std::min(f.src_start - last_src_end_, 5.0);
            }
            last_src_end_ = std::max(last_src_end_, f.src_end);
            const double behind = backlog_s(f);
            const double source = f.src_end - f.src_start;
            const double predicted = seconds_per_char_ * static_cast<double>(characters(f.text)) / engine_speed(f.prosody, 1.0f);
            pace = pacing_.speed(behind, predicted, source, gap_s_);
            pace_metric_->record(static_cast<std::uint64_t>(pace * 1e4f + 0.5f));
            log::debug("tts: utterance ", f.utterance, " would wait ", behind, " s and last ~", predicted, " s for ",
                       source, " s of source + ~", gap_s_, " s gap: speaking x", pace);
        }
        for (std::size_t i = 0; i < clauses.size(); ++i) {
            Job& job = pending_.emplace_back();
            job.speech = f;
            job.clause = clauses[i];
            job.index = static_cast<std::uint32_t>(i);
            job.last = i + 1 == clauses.size();
            job.pace = pace;
        }
        // First clause right away: first audio before the sentence ends.
        if (!pending_.empty()) {
            const Job job = std::move(pending_.front());
            pending_.pop_front();
            synthesize(job);
        }
        return;
    }
    default: return;
    }
}

void TtsStage::tick() {
    if (pending_.empty()) return;
    const Job job = std::move(pending_.front());
    pending_.pop_front();
    synthesize(job);
}

void TtsStage::close() {
    while (!pending_.empty()) tick();
}

double TtsStage::backlog_s(const Frame& speech) const {
    const AudioIo* io = ctx_->services().audio;
    double behind = io != nullptr ? io->playout_delay(speech.src_end) : 0.0;
    // Clauses not yet synthesized have not reached playback: estimate their length from the
    // recent seconds per character.
    for (const Job& j : pending_) {
        behind += seconds_per_char_ * static_cast<double>(characters(j.clause.text)) / engine_speed(j.speech.prosody, j.pace);
    }
    return behind;
}

void TtsStage::synthesize(const Job& job) {
    const auto engine = engine_.acquire();
    SynthesisRequest req;
    req.text = job.clause.text;
    req.language = job.speech.language;
    req.prosody = job.speech.prosody;
    if (job.pace != 1.0f) req.prosody.rate_pct = ((1.0f + req.prosody.rate_pct / 100.0f) * job.pace - 1.0f) * 100.0f;
    req.emphasis = job.clause.emphasis;
    req.style = &job.speech.style;
    req.utterance_final = job.last;
    // The controller's target: the source emotion plus the closed-loop correction (4.1, 5.2).
    EmotionState target = job.speech.emotion;
    target.vad = (job.speech.emotion.vad + job.speech.delta).clamped();
    target.label = nearest_label(target.vad);
    req.emotion = &target;
    if (use_voice_print_) {
        const SpeakerVoice* voice = nullptr;
        if (const auto it = voices_.find(job.speech.utterance); it != voices_.end()) {
            voice = &it->second;
        } else if (have_voice_) {
            voice = &latest_voice_;
        }
        if (voice != nullptr) {
            req.voice = &voice->print;
            req.voice_f0 = voice->f0;
        }
    }
    engine->synthesize(req, result_);
    if (const std::size_t n = characters(job.clause.text); n > 0 && result_.sample_rate > 0) {
        const double at_speed_one = static_cast<double>(result_.audio.size()) / result_.sample_rate * engine_speed(req.prosody, 1.0f);
        const double per_char = at_speed_one / static_cast<double>(n);
        seconds_per_char_ = seconds_per_char_ == 0.0 ? per_char : 0.8 * seconds_per_char_ + 0.2 * per_char;
    }
    emit_audio(job, 0);
}

void TtsStage::emit_audio(const Job& job, std::uint32_t extra_flags) {
    const std::size_t total = result_.audio.size();
    const std::size_t step = std::max<std::size_t>(chunk_samples_, 1);
    std::size_t pos = 0;
    do {
        const std::size_t n = std::min(step, total - pos);
        const bool clause_end = pos + n >= total;
        Frame& out = ctx_->make(FrameKind::SynthAudio);
        out.copy_header_from(job.speech);
        out.seq = job.index;
        out.flags = extra_flags;
        if (clause_end) out.flags |= frame_flags::kClauseEnd | (job.last ? frame_flags::kFinal : 0u);
        out.sample_rate = result_.sample_rate;
        out.audio.assign(result_.audio.begin() + static_cast<std::ptrdiff_t>(pos),
                         result_.audio.begin() + static_cast<std::ptrdiff_t>(pos + n));
        out.text = job.clause.text;
        out.language = job.speech.language;
        out.emotion = job.speech.emotion;
        out.axis_confidence = job.speech.axis_confidence;
        out.prosody = job.speech.prosody;
        if (job.index == 0 && pos == 0) {
            ctx_->services().telemetry->mark(job.speech.utterance, telemetry::Milestone::TtsFirstChunk, ctx_->now());
        }
        ctx_->emit(out);
        pos += n;
    } while (pos < total);
}

}  // namespace ee
