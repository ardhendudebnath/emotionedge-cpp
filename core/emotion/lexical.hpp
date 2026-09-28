#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "core/emotion/acoustic.hpp"

namespace ee {

/// Lexical emotion slot (blueprint 2.2 "LEXICAL: DistilRoBERTa on ASR text").
class ILexicalEmotionModel {
public:
    virtual ~ILexicalEmotionModel() = default;
    [[nodiscard]] virtual bool supports(std::string_view language) const = 0;
    [[nodiscard]] virtual ModalityEstimate estimate(std::string_view text, std::string_view language) = 0;
};

/// A small hand-authored English lexicon with phrase matching, negation, intensifiers and
/// punctuation/capitalization cues. It is the v1 stand-in for the DistilRoBERTa classifier
/// (roadmap phase 2) and abstains (invalid estimate) on languages it does not cover, so the
/// fusion gate drops it there.
class LexiconEmotionModel final : public ILexicalEmotionModel {
public:
    [[nodiscard]] bool supports(std::string_view language) const override;
    [[nodiscard]] ModalityEstimate estimate(std::string_view text, std::string_view language) override;
};

/// Lowercased word tokens; "!" and "?" become their own tokens and curly apostrophes are
/// normalized so "can’t" matches "can't".
[[nodiscard]] std::vector<std::string> tokenize_for_emotion(std::string_view text);

/// `lexical: lexicon` (default), `onnx` (a DistilRoBERTa classifier directory, `lexical_model` or
/// `lexical_model_id`; needs -DEE_WITH_ONNXRUNTIME=ON) or `none`.
[[nodiscard]] std::unique_ptr<ILexicalEmotionModel> make_lexical_model(const Params& params,
                                                                       const ModelRegistry* registry);

#if defined(EE_HAVE_ONNXRUNTIME)
[[nodiscard]] std::unique_ptr<ILexicalEmotionModel> make_onnx_lexical_model(const std::string& dir,
                                                                            const Params& params,
                                                                            const ModelRegistry* registry);
#endif

}  // namespace ee
