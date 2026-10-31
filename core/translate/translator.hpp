#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/runtime/model_registry.hpp"
#include "core/runtime/params.hpp"

namespace ee {

struct TranslationRequest {
    std::string_view source;           ///< may start with control tokens and contain <em> markup
    std::string_view source_language;  ///< ISO 639-1
    std::string_view target_language;
    std::vector<std::string> target_prefix;  ///< committed target words to keep (wait-k)
    bool want_alignment = false;
    /// Engines that stream (ITranslator::streams) stop decoding greedily once this many target
    /// words after the prefix are complete, and return them (partial). 0: translate it all.
    std::size_t stop_after_words = 0;
};

struct TranslationResult {
    std::string text;  ///< may contain <em> markup when the engine preserves it
    /// (source word, target word) pairs, when the engine can align (attention).
    std::vector<std::pair<std::uint16_t, std::uint16_t>> alignment;
    /// Stopped at stop_after_words: `text` is the prefix plus that many complete words.
    bool partial = false;
};

/// Machine-translation slot (blueprint 3.2: CTranslate2 running NLLB-200).
class ITranslator {
public:
    virtual ~ITranslator() = default;
    [[nodiscard]] virtual TranslationResult translate(const TranslationRequest& request) = 0;
    /// True once the model has been fine-tuned on the emotion control tokens (P2 LoRA).
    [[nodiscard]] virtual bool understands_control_tokens() const noexcept = 0;
    /// True if <em> markup survives translation.
    [[nodiscard]] virtual bool preserves_markup() const noexcept = 0;
    /// True if it takes <em> markup in its source (to align the emphasis), even if its output
    /// has none. Engines that do not get plain text.
    [[nodiscard]] virtual bool reads_markup() const noexcept { return preserves_markup(); }
    /// True if it honours TranslationRequest::stop_after_words.
    [[nodiscard]] virtual bool streams() const noexcept { return false; }
};

/// `engine: phrasebook` (+ `phrasebook` file) or `ct2` (+ `model`/`model_id` directory with a
/// CTranslate2 NLLB export and `sentencepiece.bpe.model`).
[[nodiscard]] std::unique_ptr<ITranslator> make_translator(const Params& params, const ModelRegistry* registry);

#if defined(EE_HAVE_CTRANSLATE2)
[[nodiscard]] std::unique_ptr<ITranslator> make_ct2_translator(const std::string& model_dir, const Params& params);
#endif

}  // namespace ee
