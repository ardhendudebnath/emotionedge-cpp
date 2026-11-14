#pragma once
// The session's language (2.1). whisper.cpp detects each utterance's language on its own, from
// the utterance's first decode, about a second of audio. Short shouts and background noise flip
// it: on RAVDESS, "Kids!" was heard as Japanese, and noise turned more English into German,
// Korean or Chinese. Every flip sends a transcript in the wrong language on to translation. The
// prior keeps the session's language unless whisper is sure of another one.
//
// Utterances transcribed in their true language (ml/README.md, "Background noise"):
// - English, without a prior: 0.93 clean, 0.80-0.89 in noise;
// - English, with the prior at switch_p 0.7: 0.97 and 0.91-0.95;
// - Hindi (synthesised): 0.51-0.54 without, 0.85-0.95 with, the session having switched to it.
// The range 0.65-0.75 is about as good; at 0.8 the Hindi sessions seldom switch.
#include <string>
#include <string_view>
#include <utility>

#include "core/runtime/config.hpp"
#include "core/runtime/params.hpp"

namespace ee {

class LanguagePrior {
public:
    /// `start`: the language the session starts in ("" = none yet; the first language heard
    /// with at least `switch_p` becomes it). `switch_p`: how sure whisper must be of another
    /// language to take over.
    LanguagePrior(std::string start, float switch_p) : session_(std::move(start)), switch_p_(switch_p) {}

    /// `language_prior`: a language code to start in, `auto` (start in none), or `off` (no prior:
    /// each utterance's own detection, as before). The asr stage defaults it to the pipeline's
    /// source language. `language_switch_p`: 0-1.
    [[nodiscard]] static bool enabled(const Params& p) { return p.str("language_prior", "auto") != "off"; }
    [[nodiscard]] static LanguagePrior from(const Params& p) {
        std::string start = p.str("language_prior", "auto");
        if (start == "auto" || start == "off") start.clear();
        const float switch_p = p.real("language_switch_p", 0.7f);
        if (switch_p < 0.0f || switch_p > 1.0f) throw ConfigError("asr language_switch_p must be 0-1");
        return {std::move(start), switch_p};
    }

    /// The language to transcribe an utterance in, given the most probable one whisper detected
    /// and its probability. `commit`: the decision may change the session's language (not for a
    /// tentative decode, which may be thrown away).
    ///
    /// A detection of another language at `switch_p` or above moves the session there. Two
    /// alternatives were replayed over the same detections (ml/README.md, "Background noise"):
    /// - Moving only when two utterances in a row agree kept English best. But whisper base
    ///   rarely names Hindi twice running, so a Hindi speaker's session seldom switched.
    /// - An easier way back to the start language changed little either way.
    [[nodiscard]] std::string choose(std::string_view best, float best_p, bool commit) {
        if (session_.empty() || best == session_) {
            if (commit && session_.empty() && best_p >= switch_p_) session_ = best;
            return std::string(best);
        }
        if (best_p < switch_p_) return session_;  // not sure enough: the session's language
        if (commit) session_ = best;
        return std::string(best);
    }

    [[nodiscard]] const std::string& session() const noexcept { return session_; }
    [[nodiscard]] float switch_p() const noexcept { return switch_p_; }

private:
    std::string session_;
    float switch_p_;
};

}  // namespace ee
