#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ee {

/// Byte-level BPE, as in GPT-2 and RoBERTa (DistilRoBERTa is the lexical emotion model, 2.2).
/// It reproduces Hugging Face `tokenizers` for these models:
///  - pre-tokenization uses the GPT-2 pattern
///    `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`;
///  - each piece's bytes map to printable characters (Ġ for a space), then BPE merges apply by rank;
///  - the ids are wrapped as `<s> ... </s>`.
/// Letter/number/space classes are exact for ASCII and Latin-1. Beyond that they are
/// approximations: common punctuation, symbol and emoji blocks count as symbols, and everything
/// else counts as a letter. This is ample for the English text the lexical model reads.
class ByteLevelBpeTokenizer {
public:
    /// Reads a Hugging Face `tokenizer.json` (BPE model with vocab + merges, ByteLevel
    /// pre-tokenizer, RoBERTa post-processor). Throws ConfigError.
    [[nodiscard]] static ByteLevelBpeTokenizer from_tokenizer_json(const std::filesystem::path& path);

    ByteLevelBpeTokenizer(std::unordered_map<std::string, std::int64_t> vocab,
                          const std::vector<std::pair<std::string, std::string>>& merges, std::int64_t bos_id,
                          std::int64_t eos_id, bool add_prefix_space = false);

    /// `<s>` + BPE ids + `</s>`, at most `max_tokens` in total (the text is truncated).
    [[nodiscard]] std::vector<std::int64_t> encode(std::string_view text, std::size_t max_tokens = 512) const;
    /// The pre-tokenizer pieces (original UTF-8 text), for tests.
    [[nodiscard]] std::vector<std::string> pre_tokenize(std::string_view text) const;
    /// BPE tokens (byte-mapped strings such as "Ġworld") of one pre-token piece, for tests.
    [[nodiscard]] std::vector<std::string> bpe(std::string_view piece) const;

    [[nodiscard]] std::size_t vocab_size() const noexcept { return vocab_.size(); }

private:
    std::unordered_map<std::string, std::int64_t> vocab_;
    std::unordered_map<std::string, int> merge_rank_;  ///< "left right" -> rank
    std::int64_t bos_id_;
    std::int64_t eos_id_;
    bool add_prefix_space_;
};

}  // namespace ee
