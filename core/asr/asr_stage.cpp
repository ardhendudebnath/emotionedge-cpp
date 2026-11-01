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
    speculation_ = SpeculationTracker(static_cast<std::size_t>(rate_ * p.integer("speculate_ms", 0) / 1000));
    speculate_pad_ = static_cast<std::size_t>(rate_ * p.integer("speculate_pad_ms", 0) / 1000);
    auto& metrics = ctx.services().telemetry->metrics();
    kept_ = &metrics.counter("ee_speculative_finals_kept_total",
                             "Final passes started in the endpoint hangover and used", {{"stage", "asr"}});
    wasted_ = &metrics.counter("ee_speculative_finals_wasted_total",
                               "Final passes started in the endpoint hangover, dropped as speech resumed", {{"stage", "asr"}});
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
        speculation_.reset();
    }
    audio_.insert(audio_.end(), f.audio.begin(), f.audio.end());

    if (f.has(frame_flags::kEndpoint)) {
        src_start_ = f.src_start;
        src_end_ = f.src_end;
        origin_ = f.t_origin;
        if (speculation_.holds()) {  // decoded in the hangover; only quiet audio came since
            kept_->inc();
            if (!early_.language.empty()) language_ = early_.language;
            emit_final(early_);
        } else {
            emit_final(transcribe(true));
        }
        speculation_.reset();
        utterance_ = 0;
        return;
    }
    switch (speculation_.on_audio(f)) {
    case SpeculationTracker::Event::Start: early_ = transcribe(true, true); return;
    case SpeculationTracker::Event::Cancel: wasted_->inc(); break;
    case SpeculationTracker::Event::None: break;
    }
    if (audio_.size() >= next_partial_at_) {
        emit_partial(transcribe(false));
        while (next_partial_at_ <= audio_.size()) next_partial_at_ += partial_interval_;
    }
}

AsrResult StreamingAsrStage::transcribe(bool final, bool early) {
    const std::shared_ptr<IAsrEngine> engine = engine_.acquire();
    AsrRequest request;
    request.audio = audio_;
    request.tentative = early;
    if (early && speculate_pad_ > 0) {
        padded_.assign(audio_.begin(), audio_.end());
        padded_.resize(audio_.size() + speculate_pad_, 0.0f);
        request.audio = padded_;
    }
    request.sample_rate = rate_;
    request.utterance_start_s = static_cast<double>(start_pos_) / rate_;
    request.final = final;
    request.language = language_hint_;

    const TimePoint t0 = Clock::now();
    AsrResult result = engine->transcribe(request);
    const double decode_s = seconds_between(t0, Clock::now());
    const double audio_s = static_cast<double>(audio_.size()) / rate_;
    if (audio_s > 0.0) ctx_->services().telemetry->record_rtf(static_cast<float>(decode_s / audio_s));
    if (!early && !result.language.empty()) language_ = result.language;  // an early final's, if it is used
    return result;
}

void StreamingAsrStage::emit_final(const AsrResult& result) {
    ctx_->services().telemetry->mark(utterance_, telemetry::Milestone::AsrFinal, ctx_->now());
    const std::vector<Word> words = agreement_.finalize(result.words);
    emit(words, static_cast<std::uint32_t>(words.size()), true);
}

void StreamingAsrStage::emit_partial(const AsrResult& result) {
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
