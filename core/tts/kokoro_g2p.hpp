#pragma once

#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace ee::kokoro {

/// The grapheme-to-phoneme front end Kokoro-82M was trained with for espeak languages
/// (misaki's EspeakG2P on top of the `phonemizer` library), ported so the C++ runtime feeds
/// Kokoro the exact phoneme strings it expects. The steps:
///  1. misaki swaps `«»` for curly quotes and `()` for `«»`;
///  2. phonemizer splits the text at punctuation runs, keeping the marks;
///  3. each chunk goes through espeak-ng (IPA, stress kept, `^` tying multi-letter phonemes), with
///     language-switch flags such as `(en)` removed;
///  4. the marks are restored around the phonemized chunks;
///  5. misaki merges tied pairs into Kokoro's single-letter symbols (`t^ʃ` -> `ʧ`), drops
///     ties and hyphens, and turns `«»` back into `()`.
/// Step 3 is a callback (espeak-ng in the runtime, a fake in tests); the rest is plain text.

/// A punctuation run and where it sits: B(egin), E(nd), I(nside), or A(lone) = the whole text.
struct PunctuationMark {
    std::string text;
    char position = 'I';
};

struct PunctuationSplit {
    std::vector<std::string> chunks;  ///< non-empty text between the marks
    std::vector<PunctuationMark> marks;
};

/// phonemizer's Punctuation.preserve for one line (default marks `;:,.!?¡¿—…"«»“”(){}[]`).
[[nodiscard]] PunctuationSplit preserve_punctuation(std::string_view line);

/// phonemizer's Punctuation.restore (word separator ' ', strip off): the first output line.
[[nodiscard]] std::string restore_punctuation(std::vector<std::string> phonemized,
                                              const std::vector<PunctuationMark>& marks);

/// phonemizer's EspeakBackend._postprocess_line for one espeak output (tie '^', stress kept,
/// language-switch flags removed): words are followed by ' '.
[[nodiscard]] std::string postprocess_espeak_line(std::string_view raw);

/// The whole EspeakG2P call. `espeak` maps one text chunk to espeak-ng's raw IPA output
/// (clauses joined by ' ', U+0361 ties).
[[nodiscard]] std::string misaki_g2p(std::string_view text,
                                     const std::function<std::string(const std::string&)>& espeak);

}  // namespace ee::kokoro
