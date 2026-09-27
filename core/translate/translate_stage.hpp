#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "core/prosody/expressivity.hpp"
#include "core/runtime/model_registry.hpp"
#include "core/runtime/stage.hpp"
#include "core/telemetry/metrics.hpp"
#include "core/translate/control_tokens.hpp"
#include "core/translate/glossary.hpp"
#include "core/translate/translator.hpp"
#include "core/translate/wait_k.hpp"

namespace ee {

/// Stage 3.2 "Emotion-Aware Translation". Final pass: prefixes the source with emotion control
/// tokens (neutral below the confidence threshold τ), marks emphasized words, locks glossary
/// terms, translates, and carries the emphasis over to target words. Drafts: translates the
/// ASR's stable prefix under a wait-k policy for live captions; the final pass keeps the
/// committed draft prefix.
class TranslateStage final : public IStage {
public:
    void open(StageContext& ctx) override;
    void process(Frame& frame) override;

private:
    enum class TokenMode { Auto, On, Off };
    void translate_final(const Frame& utterance);
    void translate_draft(const Frame& transcript);
    /// The exact string sent to the engine: control prefix (if used) + source with markup and
    /// glossary placeholders.
    [[nodiscard]] std::string prepare_input(const std::string& marked, const ControlTokens& tokens,
                                            const ITranslator& translator, Glossary::Protected& protected_text) const;

    StageContext* ctx_ = nullptr;
    HotSwap<ITranslator> translator_;
    Glossary glossary_;
    ExpressivityProfiles profiles_ = ExpressivityProfiles::defaults();
    std::string source_language_;
    std::string target_language_;
    float tau_ = 0.35f;
    float arousal_step_ = 0.01f;
    std::optional<Register> register_;
    TokenMode token_mode_ = TokenMode::Auto;
    bool drafts_ = true;
    WaitKPolicy wait_k_;
    std::uint64_t draft_utterance_ = 0;
    std::vector<std::string> committed_;
    std::uint32_t last_stable_ = 0;
    EmotionState last_emotion_;
    telemetry::Counter* glossary_misses_ = nullptr;
};

}  // namespace ee
