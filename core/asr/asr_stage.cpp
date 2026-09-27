#include "core/asr/asr_stage.hpp"

#include <algorithm>

#include "core/asr/scripted_engine.hpp"
#include "core/telemetry/telemetry.hpp"

namespace ee {

std::unique_ptr<IAsrEngine> make_asr_engine(const Params& params, const ModelRegistry* registry) {
    const std::string kind = params.str("engine", "scripted");
    if (kind == "scripted") {
        const std::string script = params.str("script");
        if (script.empty()) throw ConfigError("asr engine 'scripted' needs a 'script' file (see ScriptedAsrEngine)");
        return std::make_unique<ScriptedAsrEngine>(ScriptedAsrEngine::from_file(script));
    }
    if (kind == "whisper") {
#if defined(EE_HAVE_WHISPER)
        const std::string model = resolve_model_path(params, registry, "model");
        if (model.empty()) throw ConfigError("asr engine 'whisper' needs a 'model' path or 'model_id'");
        return make_whisper_engine(model, params);
#else
        (void)registry;
        throw ConfigError("asr engine 'whisper' needs a build with -DEE_WITH_WHISPER=ON");
#endif
    }
    throw ConfigError("unknown asr engine '" + kind + "' (scripted | whisper)");
}

void StreamingAsrStage::open(StageContext& ctx) {
    ctx_ = &ctx;
    const Params& p = ctx.params();
    rate_ = ctx.pipeline().sample_rate;
    engine_.publish(std::shared_ptr<IAsrEngine>(make_asr_engine(p, ctx.services().models)));
    agreement_ = LocalAgreement(static_cast<int>(p.integer("agreement", 2)));
    partial_interval_ = static_cast<std::size_t>(rate_ * p.integer("partial_interval_ms", 500) / 1000);
    min_partial_ = static_cast<std::size_t>(rate_ * p.integer("min_partial_ms", 1000) / 1000);
    language_hint_ = p.str("language", ctx.pipeline().source_language);
    audio_.reserve(static_cast<std::size_t>(rate_) * 20);
}

void StreamingAsrStage::process(Frame& f) {
    if (f.kind != FrameKind::Audio || f.utterance == 0) return;
    if (f.utterance != utterance_) {
        utterance_ = f.utterance;
        audio_.clear();
        agreement_.reset();
        next_partial_at_ = std::max(partial_interval_, min_partial_);
        start_pos_ = f.stream_pos;
        src_start_ = f.src_start;
        last_partial_.clear();
        language_.clear();
    }
    audio_.insert(audio_.end(), f.audio.begin(), f.audio.end());

    if (f.has(frame_flags::kEndpoint)) {
        src_start_ = f.src_start;
        src_end_ = f.src_end;
        origin_ = f.t_origin;
        decode(true);
        utterance_ = 0;
        return;
    }
    if (audio_.size() >= next_partial_at_) {
        decode(false);
        while (next_partial_at_ <= audio_.size()) next_partial_at_ += partial_interval_;
    }
}

void StreamingAsrStage::decode(bool final) {
    const std::shared_ptr<IAsrEngine> engine = engine_.acquire();
    AsrRequest request;
    request.audio = audio_;
    request.sample_rate = rate_;
    request.utterance_start_s = static_cast<double>(start_pos_) / rate_;
    request.final = final;
    request.language = language_hint_;

    const TimePoint t0 = Clock::now();
    AsrResult result = engine->transcribe(request);
    const double decode_s = seconds_between(t0, Clock::now());
    const double audio_s = static_cast<double>(audio_.size()) / rate_;
    auto* telemetry = ctx_->services().telemetry;
    if (audio_s > 0.0) telemetry->record_rtf(static_cast<float>(decode_s / audio_s));
    if (!result.language.empty()) language_ = result.language;

    if (final) {
        telemetry->mark(utterance_, telemetry::Milestone::AsrFinal, ctx_->now());
        const std::vector<Word> words = agreement_.finalize(result.words);
        emit(words, static_cast<std::uint32_t>(words.size()), true);
        return;
    }
    agreement_.update(result.words);
    const std::vector<Word> words = agreement_.current();
    const std::string text = join_words(words);
    if (text.empty() || text == last_partial_) return;  // nothing new to show
    last_partial_ = text;
    emit(words, static_cast<std::uint32_t>(agreement_.committed().size()), false);
}

void StreamingAsrStage::emit(const std::vector<Word>& words, std::uint32_t stable, bool final) {
    Frame& out = ctx_->make(FrameKind::Transcript);
    out.utterance = utterance_;
    out.stream_pos = start_pos_;
    out.src_start = src_start_;
    out.src_end = src_end_;
    out.t_origin = origin_;
    out.flags = final ? frame_flags::kFinal : 0u;
    out.words = words;
    out.text = join_words(words);
    out.stable_words = stable;
    out.language = language_.empty() ? language_hint_ : language_;
    if (out.language == "auto") out.language = ctx_->pipeline().source_language;
    ctx_->emit(out);
}

}  // namespace ee
