#include "core/tts/formant_synth.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <numbers>

#include "core/audio/dsp.hpp"
#include "core/audio/speaker.hpp"
#include "core/translate/languages.hpp"
#include "core/tts/clause_chunker.hpp"

namespace ee {

namespace {

/// Deterministic intonation pattern in [-1, 1] per syllable.
float pattern(std::uint32_t i) {
    std::uint32_t x = (i + 1) * 2654435761u;
    x ^= x >> 13;
    x *= 0x5bd1e995u;
    x ^= x >> 15;
    return static_cast<float>(x % 2001u) / 1000.0f - 1.0f;
}

/// Decodes one UTF-8 code point starting at `i` and advances `i`.
char32_t next_codepoint(std::string_view s, std::size_t& i) {
    const auto c = static_cast<unsigned char>(s[i]);
    int n = 1;
    char32_t cp = c;
    if (c >= 0xF0) {
        n = 4;
        cp = c & 0x07u;
    } else if (c >= 0xE0) {
        n = 3;
        cp = c & 0x0Fu;
    } else if (c >= 0xC0) {
        n = 2;
        cp = c & 0x1Fu;
    }
    for (int k = 1; k < n && i + static_cast<std::size_t>(k) < s.size(); ++k) {
        cp = (cp << 6) | (static_cast<unsigned char>(s[i + static_cast<std::size_t>(k)]) & 0x3Fu);
    }
    i += static_cast<std::size_t>(n);
    return cp;
}

double smoothstep(double x) {
    x = std::clamp(x, 0.0, 1.0);
    return x * x * (3.0 - 2.0 * x);
}

/// Vowel-like colouring: gentle boosts near typical first and second formants.
double formant_gain(double f) {
    const auto bump = [](double x, double center, double width) {
        const double z = (x - center) / width;
        return std::exp(-z * z);
    };
    return 1.0 + 2.0 * bump(f, 650.0, 250.0) + 1.2 * bump(f, 1150.0, 350.0) + 0.4 * bump(f, 2500.0, 500.0);
}

/// One word: continuously voiced, one pitch target per syllable.
struct WordPlan {
    double start_s = 0.0;
    double end_s = 0.0;
    std::vector<double> syllable_start;
    std::vector<double> syllable_dur;
    std::vector<double> target_st;  ///< semitones relative to the base F0
    double gain = 0.0;
    double final_fall_st = 0.0;     ///< glide over the last syllable (utterance-final word only)
};

void render_word(const WordPlan& w, double base_f0, double rolloff, double dip, double sr, std::vector<float>& audio) {
    const auto begin = static_cast<std::size_t>(w.start_s * sr);
    const auto n = static_cast<std::size_t>((w.end_s - w.start_s) * sr);
    if (n == 0 || w.target_st.empty()) return;

    const double max_st = *std::max_element(w.target_st.begin(), w.target_st.end());
    double mean_st = 0.0;
    for (double st : w.target_st) mean_st += st;
    mean_st /= static_cast<double>(w.target_st.size());
    const double f_max = base_f0 * std::pow(2.0, max_st / 12.0);
    const double f_mean = base_f0 * std::pow(2.0, mean_st / 12.0);
    const int harmonics = std::max(1, static_cast<int>(std::min(0.45 * sr, 5000.0) / f_max));
    std::vector<double> amp(static_cast<std::size_t>(harmonics) + 1, 0.0);
    double power = 0.0;
    for (int k = 1; k <= harmonics; ++k) {
        const double a = std::pow(k, -rolloff) * formant_gain(k * f_mean);
        amp[static_cast<std::size_t>(k)] = a;
        power += a * a / 2.0;
    }
    const double norm = w.gain / std::sqrt(std::max(power, 1e-12));
    const double attack = std::min(0.025, 0.3 * (w.end_s - w.start_s));
    const double release = std::min(0.040, 0.3 * (w.end_s - w.start_s));

    double phase = 0.0;
    std::size_t k = 0;
    for (std::size_t i = 0; i < n && begin + i < audio.size(); ++i) {
        const double t = w.start_s + static_cast<double>(i) / sr;
        while (k + 1 < w.syllable_start.size() && t >= w.syllable_start[k + 1]) ++k;
        const double x = (t - w.syllable_start[k]) / w.syllable_dur[k];

        // Pitch: glide from the previous syllable's target, then hold (or fall, word-finally).
        const double prev = k > 0 ? w.target_st[k - 1] : w.target_st[k];
        double st = prev + (w.target_st[k] - prev) * smoothstep(x / 0.3);
        if (k + 1 == w.target_st.size() && w.final_fall_st != 0.0) st -= w.final_fall_st * smoothstep((x - 0.3) / 0.7);
        const double f0 = base_f0 * std::pow(2.0, st / 12.0);
        phase += 2.0 * std::numbers::pi * f0 / sr;

        // Envelope: word attack/release, and an energy dip at each syllable boundary.
        const double from_start = t - w.start_s;
        const double to_end = w.end_s - t;
        double env = 1.0;
        if (from_start < attack) env *= 0.5 - 0.5 * std::cos(std::numbers::pi * from_start / attack);
        if (to_end < release) env *= 0.5 - 0.5 * std::cos(std::numbers::pi * to_end / release);
        for (std::size_t b = 1; b < w.syllable_start.size(); ++b) {
            const double half = std::min(0.015, 0.3 * w.syllable_dur[b]);
            const double d = std::abs(t - w.syllable_start[b]);
            if (d < half) env *= 1.0 - (1.0 - dip) * (0.5 + 0.5 * std::cos(std::numbers::pi * d / half));
        }

        // sin(k·phase) by the Chebyshev recurrence: s_k = 2cos(phase)·s_{k-1} - s_{k-2}.
        const double c2 = 2.0 * std::cos(phase);
        double s_prev = 0.0;
        double s_cur = std::sin(phase);
        double v = amp[1] * s_cur;
        for (int h = 2; h <= harmonics; ++h) {
            const double s_next = c2 * s_cur - s_prev;
            s_prev = s_cur;
            s_cur = s_next;
            v += amp[static_cast<std::size_t>(h)] * s_cur;
        }
        audio[begin + i] += static_cast<float>(norm * env * v);
    }
}

}  // namespace

int count_syllables(std::string_view word) {
    bool ascii = true;
    for (char c : word) ascii = ascii && static_cast<unsigned char>(c) < 0x80;
    if (ascii) {
        int groups = 0;
        bool in_vowel = false;
        std::string lower;
        for (char c : word) {
            if (std::isalpha(static_cast<unsigned char>(c)) != 0) lower += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
        for (char c : lower) {
            const bool vowel = c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u' || c == 'y';
            if (vowel && !in_vowel) ++groups;
            in_vowel = vowel;
        }
        if (groups > 1 && lower.size() > 2 && lower.back() == 'e' && lower[lower.size() - 2] != 'l') --groups;  // silent e
        return std::max(1, groups);
    }
    int bases = 0;
    int viramas = 0;
    int others = 0;
    for (std::size_t i = 0; i < word.size();) {
        const char32_t cp = next_codepoint(word, i);
        if (cp >= 0x0900 && cp <= 0x097F) {
            const bool combining = (cp >= 0x0900 && cp <= 0x0903) || cp == 0x093C || (cp >= 0x093E && cp <= 0x094F) ||
                                   (cp >= 0x0951 && cp <= 0x0957) || cp == 0x0962 || cp == 0x0963;
            if (cp == 0x094D) ++viramas;
            if (!combining) ++bases;
        } else if (cp > 0x7F || std::isalnum(static_cast<int>(cp)) != 0) {
            ++others;
        }
    }
    const int devanagari = bases - viramas;
    return std::max(1, devanagari + (others + 1) / 2);
}

FormantSynth::FormantSynth(FormantVoice voice) : voice_(voice) {}

void FormantSynth::synthesize(const SynthesisRequest& req, SynthesisResult& out) {
    const ProsodyTargets& p = req.prosody;
    const double sr = voice_.sample_rate;
    const double rate = std::clamp(1.0 + p.rate_pct / 100.0, 0.5, 2.0);
    const double mean_st = 12.0 * std::log2(std::max(0.2, 1.0 + p.pitch_pct / 100.0));
    const double range = voice_.range_st * std::max(0.0, 1.0 + p.range_pct / 100.0);
    const double hesitation = std::max(0.0f, p.hesitation);
    // Louder and tenser voices have flatter spectra (vocal effort).
    const double rolloff = std::clamp(voice_.rolloff - 0.5 * p.tension - 0.05 * p.energy_db, 0.7, 2.4);
    double base_f0 = voice_.base_f0_hz;
    if (req.voice != nullptr) {
        if (const float f0 = req.voice_f0; f0 > 0.0f) base_f0 = f0;
    }

    const std::vector<std::string> words = split_words(req.text, req.language);
    int total = 0;
    for (const std::string& w : words) total += count_syllables(w);

    std::vector<WordPlan> plans;
    out.words.clear();
    double t = 0.02;
    int index = 0;
    for (std::size_t w = 0; w < words.size(); ++w) {
        const bool emphasized = std::find(req.emphasis.begin(), req.emphasis.end(), static_cast<std::uint16_t>(w)) != req.emphasis.end();
        if (w > 0) t += voice_.word_gap_s * hesitation;
        if (emphasized) t += p.pause_ms / 1000.0;  // pre-pause before the emphasized word
        WordPlan plan;
        plan.start_s = t;
        const int syllables = count_syllables(words[w]);
        for (int k = 0; k < syllables; ++k, ++index) {
            const double dur = voice_.syllable_s / rate * (emphasized ? 1.15 : 1.0);
            const double progress = total > 1 ? static_cast<double>(index) / (total - 1) : 0.5;
            double st = mean_st + voice_.declination_st * (0.5 - progress) + range * pattern(static_cast<std::uint32_t>(index));
            if (emphasized && k == 0) st += 2.0 * p.accent;  // local pitch accent
            plan.syllable_start.push_back(t);
            plan.syllable_dur.push_back(dur);
            plan.target_st.push_back(st);
            t += dur;
        }
        plan.end_s = t;
        plan.gain = db_to_gain(voice_.level_dbfs + p.energy_db + (emphasized ? voice_.emphasis_db : 0.0f));
        if (req.utterance_final && w + 1 == words.size()) {
            plan.final_fall_st = voice_.final_fall_st + 3.0 * p.final_fall;  // + falls, - rises
        }
        out.words.push_back({words[w], static_cast<float>(plan.start_s), static_cast<float>(plan.end_s), 1.0f});
        plans.push_back(std::move(plan));
        if (w + 1 < words.size() && ends_clause(words[w])) t += voice_.clause_gap_s * hesitation;
    }
    t += 0.06;

    out.sample_rate = voice_.sample_rate;
    out.audio.assign(static_cast<std::size_t>(t * sr), 0.0f);
    for (const WordPlan& plan : plans) render_word(plan, base_f0, rolloff, voice_.syllable_dip, sr, out.audio);
    for (float& v : out.audio) v = soft_limit(v, 0.95f);
}

}  // namespace ee
