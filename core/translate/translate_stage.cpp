#include "core/translate/translate_stage.hpp"

#include "core/runtime/log.hpp"
#include "core/telemetry/telemetry.hpp"
#include "core/translate/emphasis.hpp"
#include "core/translate/languages.hpp"
#include "core/translate/phrasebook.hpp"

namespace ee {

std::unique_ptr<ITranslator> make_translator(const Params& params, const ModelRegistry* registry) {
    const std::string kind = params.str("engine", "phrasebook");
    if (kind == "phrasebook") {
        const std::string path = params.str("phrasebook");
        if (path.empty()) return std::make_unique<PhrasebookTranslator>();  // pseudo-translations only
        return std::make_unique<PhrasebookTranslator>(PhrasebookTranslator::load(path));
    }
    if (kind == "ct2") {
#if defined(EE_HAVE_CTRANSLATE2)
        const std::string model = resolve_model_path(params, registry, "model");
        if (model.empty()) throw ConfigError("translate engine 'ct2' needs a 'model' directory or 'model_id'");
        return make_ct2_translator(model, params);
#else
        (void)registry;
        throw ConfigError("translate engine 'ct2' needs a build with -DEE_WITH_CTRANSLATE2=ON");
#endif
    }
    throw ConfigError("unknown translate engine '" + kind + "' (phrasebook | ct2)");
}

void TranslateStage::open(StageContext& ctx) {
    ctx_ = &ctx;
    const Params& p = ctx.params();
    translator_.publish(std::shared_ptr<ITranslator>(make_translator(p, ctx.services().models)));
    source_language_ = p.str("source_language", ctx.pipeline().source_language);
    target_language_ = p.str("target_language", ctx.pipeline().target_language);
    if (const std::string path = p.str("expressivity"); !path.empty()) profiles_ = ExpressivityProfiles::load(path);
    if (const std::string path = p.str("glossary"); !path.empty()) glossary_ = Glossary::load(path);
    tau_ = p.real("neutral_threshold", tau_);
    arousal_step_ = p.real("arousal_step", arousal_step_);
    if (const std::string reg = p.str("register", "auto"); reg != "auto") {
        register_ = parse_register(reg);
        if (!register_) throw ConfigError("translate register must be auto, casual, formal or neutral");
    }
    const std::string tokens = p.str("control_tokens", "auto");
    if (tokens != "auto" && tokens != "on" && tokens != "off") {
        throw ConfigError("translate control_tokens must be auto, on or off (got '" + tokens + "')");
    }
    token_mode_ = tokens == "on" ? TokenMode::On : tokens == "off" ? TokenMode::Off : TokenMode::Auto;
    drafts_ = p.flag("drafts", true);
    wait_k_ = WaitKPolicy(static_cast<int>(p.integer("wait_k", 3)),
                          p.real("length_ratio", profiles_.get(target_language_).length_ratio));
    glossary_misses_ = &ctx.services().telemetry->metrics().counter(
        "ee_glossary_misses_total", "Glossary placeholders the MT engine dropped");
}

void TranslateStage::process(Frame& f) {
    if (f.kind == FrameKind::Utterance) {
        translate_final(f);
    } else if (f.kind == FrameKind::Transcript && !f.is_final()) {
        translate_draft(f);
    }
}

std::string TranslateStage::prepare_input(const std::string& marked, const ControlTokens& tokens,
                                          const ITranslator& translator, Glossary::Protected& protected_text) const {
    protected_text = glossary_.protect(marked);
    std::string body = protected_text.text;
    if (!translator.preserves_markup()) body = parse_markup(body, source_language_).plain;
    const bool use_tokens =
        token_mode_ == TokenMode::On || (token_mode_ == TokenMode::Auto && translator.understands_control_tokens());
    return use_tokens ? format_control_prefix(tokens, arousal_step_) + " " + body : body;
}

void TranslateStage::translate_final(const Frame& u) {
    if (u.text.empty()) return;
    const std::shared_ptr<ITranslator> translator = translator_.acquire();
    const Register reg = register_.value_or(profiles_.get(target_language_).reg);
    const ControlTokens tokens = make_control_tokens(u.emotion, reg, tau_, arousal_step_);

    std::vector<std::string> source_words;
    for (const Word& w : u.words) source_words.push_back(w.text);
    if (source_words.empty()) source_words = split_words(u.text, source_language_);
    const std::string marked = apply_emphasis_markup(source_words, u.emphasis, source_language_);

    Glossary::Protected protected_text;
    const std::string input = prepare_input(marked, tokens, *translator, protected_text);
    TranslationRequest request;
    request.source = input;
    request.source_language = u.language.empty() ? source_language_ : u.language;
    request.target_language = target_language_;
    if (draft_utterance_ == u.utterance) request.target_prefix = committed_;  // "prefix kept"
    request.want_alignment = !u.emphasis.empty();
    const TranslationResult result = translator->translate(request);

    std::uint32_t missing = 0;
    const std::string restored = glossary_.restore(remove_control_tokens(result.text), protected_text, &missing);
    if (missing > 0) glossary_misses_->inc(missing);
    const Markup target = parse_markup(restored, target_language_);
    const std::size_t target_words = split_words(target.plain, target_language_).size();

    Frame& out = ctx_->make(FrameKind::Translation);
    out.copy_header_from(u);
    out.flags = frame_flags::kFinal;
    out.text = target.plain;
    out.language = target_language_;
    out.detail = input;
    out.emotion = u.emotion;
    out.axis_confidence = u.axis_confidence;
    out.emphasis = project_emphasis(u.emphasis, source_words.size(), target_words, target.emphasis, result.alignment);
    ctx_->services().telemetry->mark(u.utterance, telemetry::Milestone::MtFinal, ctx_->now());
    ctx_->emit(out);

    last_emotion_ = u.emotion;
    draft_utterance_ = 0;
    committed_.clear();
    last_stable_ = 0;
}

void TranslateStage::translate_draft(const Frame& t) {
    if (!drafts_ || t.utterance == 0) return;
    if (t.utterance != draft_utterance_) {
        draft_utterance_ = t.utterance;
        committed_.clear();
        last_stable_ = 0;
    }
    const std::uint32_t stable = std::min<std::uint32_t>(t.stable_words, static_cast<std::uint32_t>(t.words.size()));
    if (stable == 0 || stable == last_stable_) return;
    last_stable_ = stable;
    if (wait_k_.target_budget(stable) <= committed_.size()) return;  // wait-k: not allowed to write yet

    const std::shared_ptr<ITranslator> translator = translator_.acquire();
    std::vector<std::string> source_words;
    for (std::uint32_t i = 0; i < stable; ++i) source_words.push_back(t.words[i].text);
    const Register reg = register_.value_or(profiles_.get(target_language_).reg);
    Glossary::Protected protected_text;
    const std::string input = prepare_input(join_text(source_words, source_language_),
                                            make_control_tokens(last_emotion_, reg, tau_, arousal_step_), *translator,
                                            protected_text);
    TranslationRequest request;
    request.source = input;
    request.source_language = t.language.empty() ? source_language_ : t.language;
    request.target_language = target_language_;
    request.target_prefix = committed_;
    const TranslationResult result = translator->translate(request);
    const Markup draft =
        parse_markup(glossary_.restore(remove_control_tokens(result.text), protected_text), target_language_);
    if (wait_k_.extend(committed_, split_words(draft.plain, target_language_), stable) == 0) return;

    Frame& out = ctx_->make(FrameKind::Translation);
    out.copy_header_from(t);
    out.text = join_text(committed_, target_language_);
    out.language = target_language_;
    out.detail = input;
    out.stable_words = static_cast<std::uint32_t>(committed_.size());
    out.emotion = last_emotion_;
    ctx_->emit(out);
}

}  // namespace ee
