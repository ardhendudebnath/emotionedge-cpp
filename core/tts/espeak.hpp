#pragma once

// espeak-ng keeps process-wide state, so every phonemizing engine (Piper, Kokoro) goes through
// these helpers. Compiled only with -DEE_WITH_PIPER=ON (which brings espeak-ng).
#if !defined(EE_HAVE_PIPER)
#error "core/tts/espeak.hpp needs a build with -DEE_WITH_PIPER=ON"
#endif

#include <mutex>
#include <string>

namespace ee::espeak {

/// Held around every espeak-ng call.
[[nodiscard]] std::mutex& mutex();

/// Initializes espeak-ng (idempotent). `data_dir` is the directory containing espeak-ng-data,
/// or empty for the library default. Throws ConfigError on failure.
void initialize(const std::string& data_dir);

/// espeak_TextToPhonemes over every clause of `text` in `voice`, joined by ' ', as IPA. With
/// `tie`, multi-letter phonemes are joined by U+0361 (what phonemizer asks for); without it the
/// IPA is plain (what Piper voices expect). Takes the mutex itself.
[[nodiscard]] std::string text_to_phonemes(const std::string& text, const std::string& voice, bool tie);

}  // namespace ee::espeak
