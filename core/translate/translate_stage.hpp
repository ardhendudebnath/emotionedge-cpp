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
///
/// Streaming (`stream_words` > 0, with an engine that streams): the final pass first decodes
/// greedily up to `stream_words` words (at least one past a committed draft prefix) and sends
/// them ahead (frame_flags::kLeading), with their emphasis as the engine aligned it, so speech
/// can start on them; it then translates the rest with them as its prefix.
class TranslateStage final : public IStage {
public:
    void open(StageContext& ctx) override;
    void process(Frame& frame) override;

private:
    enum class TokenMode { Auto, On, Off };
    void translate_final(const Frame& utterance);
    void translate_draft(const Frame& transcript);
    /// The leading words of `utterance`'s translation (engine output, glossary placeholders in),
    /// with the indices of those emphasized.
    void emit_leading(const Frame& utterance, const std::vector<std::string>& words, std::vector<std::uint16_t> emphasis,
                      const std::string& input, const Glossary::Protected& protected_text, std::size_t source_words);
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
    std::size_t stream_words_ = 0;
    float length_ratio_ = 1.0f;  ///< target words per source word (expressivity profile)
    float word_ratio_ = 1.0f;    ///< the same, as this session's final translations ran (moving average)
    WaitKPolicy wait_k_;
    std::uint64_t draft_utterance_ = 0;
    std::vector<std::string> committed_;
    std::uint32_t last_stable_ = 0;
    EmotionState last_emotion_;
    telemetry::Counter* glossary_misses_ = nullptr;
    telemetry::Counter* streamed_ = nullptr;
};

}  // namespace ee
