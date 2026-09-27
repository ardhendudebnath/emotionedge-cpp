#include "core/tts/tts_stage.hpp"

#include <algorithm>

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
    throw ConfigError("unknown tts engine '" + kind + "' (formant | piper; StyleTTS2 arrives with roadmap phase 3)");
}

void TtsStage::open(StageContext& ctx) {
    ctx_ = &ctx;
    const Params& p = ctx.params();
    engine_.publish(std::shared_ptr<ITtsEngine>(make_tts_engine(p, ctx.services().models)));
    const auto engine = engine_.acquire();
    chunker_.min_first_chars = static_cast<std::size_t>(p.integer("min_first_chars", 8));
    chunker_.min_chars = static_cast<std::size_t>(p.integer("min_chars", 24));
    chunker_.max_chars = static_cast<std::size_t>(p.integer("max_chars", 140));
    chunk_samples_ = static_cast<std::size_t>(engine->sample_rate() * p.integer("chunk_ms", 100) / 1000);
    use_voice_print_ = p.flag("use_voice_print", true);

    if (p.flag("calibration", true)) {
        // Neutral reference render of this voice, for 5.2's baseline. Playback ignores it.
        const std::string text = p.str("calibration_text", "The quick brown fox jumps over the lazy dog, and then it rests.");
        SynthesisRequest req;
        req.text = text;
        req.language = ctx.pipeline().target_language;
        engine->synthesize(req, result_);
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
        voices_[f.utterance] = f.voice;
        latest_voice_ = f.voice;
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
        for (std::size_t i = 0; i < clauses.size(); ++i) {
            Job& job = pending_.emplace_back();
            job.speech = f;
            job.clause = clauses[i];
            job.index = static_cast<std::uint32_t>(i);
            job.last = i + 1 == clauses.size();
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

void TtsStage::synthesize(const Job& job) {
    const auto engine = engine_.acquire();
    SynthesisRequest req;
    req.text = job.clause.text;
    req.language = job.speech.language;
    req.prosody = job.speech.prosody;
    req.emphasis = job.clause.emphasis;
    req.style = &job.speech.style;
    req.utterance_final = job.last;
    if (use_voice_print_) {
        if (const auto it = voices_.find(job.speech.utterance); it != voices_.end()) {
            req.voice = &it->second;
        } else if (have_voice_) {
            req.voice = &latest_voice_;
        }
    }
    engine->synthesize(req, result_);
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
