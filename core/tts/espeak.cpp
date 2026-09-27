#include "core/tts/espeak.hpp"

#include <espeak-ng/speak_lib.h>

#include "core/runtime/params.hpp"

namespace ee::espeak {

std::mutex& mutex() {
    static std::mutex m;
    return m;
}

void initialize(const std::string& data_dir) {
    std::lock_guard lock(mutex());
    static bool done = false;
    static std::string done_with;
    if (done && done_with == data_dir) return;
    if (espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 0, data_dir.empty() ? nullptr : data_dir.c_str(), 0) < 0) {
        throw ConfigError("cannot initialize espeak-ng (set espeak_data to the directory holding espeak-ng-data)");
    }
    done = true;
    done_with = data_dir;
}

std::string text_to_phonemes(const std::string& text, const std::string& voice, bool tie) {
    // phonememode: bit 1 = IPA; bit 7 = bits 8-23 hold a tie character for multi-letter phonemes.
    const int mode = tie ? (espeakPHONEMES_IPA | espeakPHONEMES_TIE | (0x0361 << 8)) : espeakPHONEMES_IPA;
    std::lock_guard lock(mutex());
    if (espeak_SetVoiceByName(voice.c_str()) != EE_OK) throw ConfigError("espeak-ng has no voice '" + voice + "'");
    std::string out;
    const void* cursor = text.c_str();
    while (cursor != nullptr) {
        const char* clause = espeak_TextToPhonemes(&cursor, espeakCHARS_UTF8, mode);
        if (clause == nullptr) break;
        if (*clause == '\0') continue;
        if (!out.empty()) out += ' ';
        out += clause;
    }
    return out;
}

}  // namespace ee::espeak
