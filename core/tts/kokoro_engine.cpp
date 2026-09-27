// Kokoro-82M (StyleTTS2 family, 24 kHz) through ONNX Runtime: the phase-3 expressive TTS
// (blueprint 4.2). Compiled only with -DEE_WITH_PIPER=ON, which brings ONNX Runtime and
// espeak-ng.
//
// A voice directory comes from ml/export/export_kokoro_onnx.py:
//   model.onnx     phoneme ids + style row + prosody controls -> 24 kHz audio, frames per token
//   config.json    phoneme vocabulary
//   voices/*.bin   voicepacks, float32 [510, 256]; the row is chosen by phoneme count
//   voices.json    per voice: gender, ECAPA-TDNN voice print, median F0
//
// The emotion controller's plan (4.1) drives the model's own prosody:
// - rate -> speed;
// - pitch mean -> a shift of the predicted F0;
// - pitch range -> F0 scaled about its mean;
// - the final contour -> a fall or rise over the clause end;
// - emphasis -> longer duration and a pitch accent on the word's phonemes, and a longer gap
//   before it;
// - hesitation -> the length of word gaps.
// Energy is output gain. Kokoro cannot clone a voice, so the speaker's voice print picks the
// nearest of the pack's voices, by ECAPA cosine or by median F0.
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include "core/audio/dsp.hpp"
#include "core/audio/speaker.hpp"
#include "core/runtime/log.hpp"
#include "core/runtime/onnx.hpp"
#include "core/translate/languages.hpp"
#include "core/tts/espeak.hpp"
#include "core/tts/kokoro_g2p.hpp"
#include "core/tts/tts_engine.hpp"

namespace ee {

namespace {

namespace fs = std::filesystem;

constexpr int kSampleRate = 24000;
constexpr std::size_t kStyleDim = 256;
constexpr std::size_t kPackRows = 510;  // Kokoro's context: at most 510 phonemes per call

std::vector<std::string> codepoints(std::string_view s) {
    std::vector<std::string> out;
    for (std::size_t i = 0; i < s.size();) {
        const auto c = static_cast<unsigned char>(s[i]);
        const std::size_t n = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
        out.emplace_back(s.substr(i, n));
        i += n;
    }
    return out;
}

nlohmann::json read_json(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw ConfigError("Kokoro voice file not found: " + path.string());
    return nlohmann::json::parse(in);
}

struct KokoroVoice {
    std::string name;
    std::string gender;
    std::vector<float> ecapa;
    float median_f0 = 0.0f;
    std::vector<float> pack;  // [510 x 256], loaded on first use
};

class KokoroEngine final : public ITtsEngine {
public:
    KokoroEngine(const std::string& dir, const Params& params, const ModelRegistry* registry)
        : dir_(dir),
          session_(onnx::shared_session((dir_ / "model.onnx").string(), onnx::session_config(params, registry))),
          espeak_voice_(params.str("espeak_voice", "hi")),
          match_(params.str("voice_match", "pitch")),
          controls_(params.flag("prosody_controls", true)),
          trim_(params.flag("trim_silence", true)) {
        // Keep the documents alive for the loops: items() refers into them.
        const nlohmann::json config = read_json(dir_ / "config.json");
        for (const auto& [symbol, id] : config.at("vocab").items()) {
            vocab_[symbol] = id.get<std::int64_t>();
        }
        const nlohmann::json voices = read_json(dir_ / "voices.json");
        for (const auto& [name, v] : voices.items()) {
            voices_.push_back({name, v.value("gender", std::string()), v.value("ecapa", std::vector<float>{}),
                               v.value("median_f0", 0.0f), {}});
        }
        if (voices_.empty()) throw ConfigError("Kokoro voice pack '" + dir + "' lists no voices");
        // Learned per-emotion style offsets (ml/train/learn_style_offsets.py), if present.
        // style_emotions limits which offsets apply: an offset that only fools the training
        // judge (ml/eval/eval_style_offsets.py) should stay off.
        style_strength_ = params.real("style_strength", 1.0f);
        const std::vector<std::string> allowed = params.list("style_emotions");
        const fs::path path = dir_ / params.str("style_offsets", "style_offsets.json");
        if (style_strength_ > 0.0f && fs::exists(path)) {
            const nlohmann::json offsets = read_json(path);
            for (const auto& [label, values] : offsets.at("offsets").items()) {
                const auto parsed = parse_emotion_label(label);
                auto vec = values.get<std::vector<float>>();
                if (!parsed || vec.size() != kStyleDim) throw ConfigError("bad style offset '" + label + "' in " + path.string());
                if (!allowed.empty() && std::find(allowed.begin(), allowed.end(), label) == allowed.end()) continue;
                offsets_[*parsed] = std::move(vec);
            }
        }
        if (match_ != "pitch" && match_ != "ecapa" && match_ != "off") {
            throw ConfigError("tts voice_match must be pitch, ecapa or off");
        }
        const std::string voice = params.str("voice", "auto");
        fixed_ = voice != "auto";
        current_ = find(fixed_ ? voice : params.str("default_voice", voices_.front().name));
        espeak::initialize(params.str("espeak_data"));
    }

    int sample_rate() const noexcept override { return kSampleRate; }

    void synthesize(const SynthesisRequest& req, SynthesisResult& out) override {
        out.sample_rate = kSampleRate;
        out.audio.clear();
        out.words.clear();
        if (!fixed_ && !matched_ && match_ != "off" && req.voice != nullptr) match_voice(*req.voice, req.voice_f0);

        const std::string ps = kokoro::misaki_g2p(
            req.text, [this](const std::string& chunk) { return espeak::text_to_phonemes(chunk, espeak_voice_, true); });
        std::vector<std::string> symbols = codepoints(ps);
        if (symbols.size() > kPackRows) symbols.resize(kPackRows);
        if (symbols.empty()) return;

        // Token ids ([0] + symbols in the vocabulary + [0]) and the phoneme word of each token.
        ids_.assign(1, 0);
        std::vector<int> token_word{-1};
        int word = 0;
        for (const std::string& s : symbols) {
            if (s == " ") ++word;
            const auto it = vocab_.find(s);
            if (it == vocab_.end()) continue;
            ids_.push_back(it->second);
            token_word.push_back(s == " " ? -1 : word);
        }
        ids_.push_back(0);
        token_word.push_back(-1);
        const int phoneme_words = word + 1;
        const std::size_t t = ids_.size();

        // prosody_controls: false renders Kokoro's own prosody (the A/B baseline for 4.1).
        static const ProsodyTargets kNeutral{};
        const ProsodyTargets& p = controls_ ? req.prosody : kNeutral;
        const std::vector<std::uint16_t> no_emphasis;
        const std::vector<std::uint16_t>& emphasis = controls_ ? req.emphasis : no_emphasis;
        dur_.assign(t, 1.0f);
        accent_.assign(t, 1.0f);
        const float hesitation = std::clamp(p.hesitation, 0.5f, 2.0f);
        for (std::size_t k = 0; k < t; ++k) {
            if (token_word[k] < 0 && k > 0 && k + 1 < t) dur_[k] = hesitation;  // word gaps
        }
        // Emphasized text words -> phoneme words (numbers and symbols can change the count).
        const std::size_t text_words = std::max<std::size_t>(1, split_words(req.text, req.language).size());
        for (std::uint16_t w : emphasis) {
            const int target = text_words == static_cast<std::size_t>(phoneme_words) || text_words == 1
                                   ? static_cast<int>(w)
                                   : static_cast<int>(std::lround(static_cast<double>(w) * (phoneme_words - 1) /
                                                                  static_cast<double>(text_words - 1)));
            bool gap_done = false;
            for (std::size_t k = 1; k + 1 < t; ++k) {
                if (token_word[k] != target) continue;
                dur_[k] *= 1.12f;
                accent_[k] = 1.0f + 0.25f * (std::max(1.0f, p.accent) - 1.0f);
                if (!gap_done && p.pause_ms > 0.0f && token_word[k - 1] < 0 && k > 1) {
                    dur_[k - 1] *= std::min(4.0f, 1.0f + p.pause_ms / 50.0f);  // pre-pause
                }
                gap_done = true;
            }
        }
        const float speed = std::clamp(1.0f + p.rate_pct / 100.0f, 0.5f, 2.0f);
        const float pitch_st = 12.0f * std::log2(std::max(0.25f, 1.0f + p.pitch_pct / 100.0f));
        const float range = std::clamp(1.0f + p.range_pct / 100.0f, 0.2f, 3.0f);
        const float fall = req.utterance_final ? std::clamp(p.final_fall, -1.0f, 1.0f) : 0.0f;

        KokoroVoice& voice = voices_[current_];
        load_pack(voice);
        const float* row = voice.pack.data() + (symbols.size() - 1) * kStyleDim;
        style_.assign(row, row + kStyleDim);
        if (controls_) apply_style_offset(req.emotion);
        const float* style = style_.data();

        const std::array<std::int64_t, 2> ids_shape{1, static_cast<std::int64_t>(t)};
        const std::array<std::int64_t, 2> style_shape{1, static_cast<std::int64_t>(kStyleDim)};
        const std::array<std::int64_t, 1> token_shape{static_cast<std::int64_t>(t)};
        const std::array<std::int64_t, 1> scalar_shape{1};
        std::array<float, 4> scalars{speed, pitch_st, range, fall};
        std::array<Ort::Value, 8> inputs{
            Ort::Value::CreateTensor<std::int64_t>(memory_, ids_.data(), t, ids_shape.data(), 2),
            Ort::Value::CreateTensor<float>(memory_, const_cast<float*>(style), kStyleDim, style_shape.data(), 2),
            Ort::Value::CreateTensor<float>(memory_, &scalars[0], 1, scalar_shape.data(), 1),
            Ort::Value::CreateTensor<float>(memory_, dur_.data(), t, token_shape.data(), 1),
            Ort::Value::CreateTensor<float>(memory_, accent_.data(), t, token_shape.data(), 1),
            Ort::Value::CreateTensor<float>(memory_, &scalars[1], 1, scalar_shape.data(), 1),
            Ort::Value::CreateTensor<float>(memory_, &scalars[2], 1, scalar_shape.data(), 1),
            Ort::Value::CreateTensor<float>(memory_, &scalars[3], 1, scalar_shape.data(), 1)};
        static constexpr std::array<const char*, 8> kInputs{"input_ids", "style", "speed",    "dur_scale",
                                                            "accent",    "pitch_st", "range", "fall"};
        static constexpr std::array<const char*, 2> kOutputs{"audio", "durations"};
        auto result = session_->Run(Ort::RunOptions{nullptr}, kInputs.data(), inputs.data(), inputs.size(),
                                    kOutputs.data(), kOutputs.size());

        const float* audio = result[0].GetTensorData<float>();
        const std::size_t samples = result[0].GetTensorTypeAndShapeInfo().GetElementCount();
        const float gain = db_to_gain(p.energy_db);
        // Kokoro renders ~0.4-0.5 s of near-silence around every clause. Shortening its boundary
        // tokens instead costs intelligibility (Whisper CER 0.19 -> 0.31), so the audio is trimmed:
        // the first chunk then carries speech, and split clauses do not drift apart.
        std::size_t keep_begin = 0;
        std::size_t keep_end = samples;
        if (trim_) {
            float peak = 0.0f;
            for (std::size_t i = 0; i < samples; ++i) peak = std::max(peak, std::abs(audio[i]));
            const float threshold = std::max(peak * 0.01f, 1e-4f);  // -40 dB of the clause peak
            std::size_t first = 0;
            while (first < samples && std::abs(audio[first]) < threshold) ++first;
            std::size_t last = samples;
            while (last > first && std::abs(audio[last - 1]) < threshold) --last;
            const std::size_t lead = kSampleRate * 20 / 1000;
            const std::size_t tail = kSampleRate * (req.utterance_final ? 120 : 60) / 1000;
            keep_begin = first > lead ? first - lead : 0;
            keep_end = std::min(samples, last + tail);
            if (keep_end <= keep_begin) keep_begin = keep_end = 0;
        }
        out.audio.resize(keep_end - keep_begin);
        for (std::size_t i = keep_begin; i < keep_end; ++i) {
            out.audio[i - keep_begin] = std::clamp(audio[i] * gain, -1.0f, 1.0f);
        }
        const double trimmed_s = static_cast<double>(keep_begin) / kSampleRate;

        // Word timings from the frames per token, when the phoneme words match the text words.
        const std::int64_t* frames = result[1].GetTensorData<std::int64_t>();
        const std::size_t tokens = result[1].GetTensorTypeAndShapeInfo().GetElementCount();
        std::int64_t total = 0;
        for (std::size_t k = 0; k < tokens; ++k) total += frames[k];
        const std::vector<std::string> words = split_words(req.text, req.language);
        if (total > 0 && tokens == t && words.size() == static_cast<std::size_t>(phoneme_words)) {
            const double seconds_per_frame = static_cast<double>(samples) / static_cast<double>(total) / kSampleRate;
            std::vector<std::pair<double, double>> spans(words.size(), {-1.0, -1.0});
            double now = 0.0;
            for (std::size_t k = 0; k < t; ++k) {
                const double end = now + static_cast<double>(frames[k]) * seconds_per_frame;
                if (token_word[k] >= 0) {
                    auto& span = spans[static_cast<std::size_t>(token_word[k])];
                    if (span.first < 0.0) span.first = now;
                    span.second = end;
                }
                now = end;
            }
            for (std::size_t w = 0; w < words.size(); ++w) {
                out.words.push_back({words[w], static_cast<float>(std::max(0.0, spans[w].first - trimmed_s)),
                                     static_cast<float>(std::max(0.0, spans[w].second - trimmed_s)), 1.0f});
            }
        }
    }

private:
    // style += strength · Δ_label. Strength is how far the target V·A·D point reaches toward
    // its label's prototype (projection, capped at 1.25), gated by confidence: none below 0.35,
    // full from 0.75.
    void apply_style_offset(const EmotionState* emotion) {
        if (emotion == nullptr || offsets_.empty()) return;
        const auto it = offsets_.find(emotion->label);
        if (it == offsets_.end()) return;
        const Vad proto = prototype(emotion->label);
        const float proto_sq = proto.v * proto.v + proto.a * proto.a + proto.d * proto.d;
        if (proto_sq <= 0.0f) return;
        const float along = (emotion->vad.v * proto.v + emotion->vad.a * proto.a + emotion->vad.d * proto.d) / proto_sq;
        const float gate = std::clamp((emotion->confidence - 0.35f) / 0.4f, 0.0f, 1.0f);
        const float strength = std::clamp(along, 0.0f, 1.25f) * gate * style_strength_;
        for (std::size_t k = 0; k < kStyleDim; ++k) style_[k] += strength * it->second[k];
    }

    std::size_t find(const std::string& name) const {
        for (std::size_t i = 0; i < voices_.size(); ++i) {
            if (voices_[i].name == name) return i;
        }
        throw ConfigError("Kokoro voice pack '" + dir_.string() + "' has no voice '" + name + "'");
    }

    // Picks the pack voice closest to the speaker, once: switching voices mid-conversation
    // would sound like a different person. `pitch` (default) compares median F0 and gets the
    // gender right for 23/24 RAVDESS actors. `ecapa` compares voice prints by cosine, which only
    // works within a domain: against Kokoro's synthetic Hindi voices it gets 11/24.
    void match_voice(const SpeakerEmbedding& print, float speaker_f0) {
        std::size_t best = current_;
        if (match_ == "ecapa") {
            double best_score = -2.0;
            for (std::size_t i = 0; i < voices_.size(); ++i) {
                if (voices_[i].ecapa.size() != print.size()) continue;
                double dot = 0.0, a = 0.0, b = 0.0;
                for (std::size_t k = 0; k < print.size(); ++k) {
                    dot += static_cast<double>(print[k]) * voices_[i].ecapa[k];
                    a += static_cast<double>(print[k]) * print[k];
                    b += static_cast<double>(voices_[i].ecapa[k]) * voices_[i].ecapa[k];
                }
                const double score = a > 0.0 && b > 0.0 ? dot / std::sqrt(a * b) : -2.0;
                if (score > best_score) {
                    best_score = score;
                    best = i;
                }
            }
        } else {
            const float f0 = speaker_f0;
            if (f0 <= 0.0f) return;  // no pitch measured yet; try again with the next print
            float best_gap = std::numeric_limits<float>::max();
            for (std::size_t i = 0; i < voices_.size(); ++i) {
                const float gap = std::abs(std::log2(std::max(1.0f, voices_[i].median_f0) / f0));
                if (gap < best_gap) {
                    best_gap = gap;
                    best = i;
                }
            }
        }
        current_ = best;
        matched_ = true;
        log::info("tts: speaker matched to Kokoro voice ", voices_[best].name, " (", voices_[best].gender, ")");
    }

    void load_pack(KokoroVoice& voice) const {
        if (!voice.pack.empty()) return;
        const fs::path path = dir_ / "voices" / (voice.name + ".bin");
        std::ifstream in(path, std::ios::binary);
        voice.pack.resize(kPackRows * kStyleDim);
        if (!in.read(reinterpret_cast<char*>(voice.pack.data()),
                     static_cast<std::streamsize>(voice.pack.size() * sizeof(float)))) {
            voice.pack.clear();
            throw ConfigError("Kokoro voicepack unreadable or short: " + path.string());
        }
    }

    fs::path dir_;
    std::shared_ptr<Ort::Session> session_;
    Ort::MemoryInfo memory_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::string espeak_voice_;
    std::string match_;
    bool controls_ = true;
    bool trim_ = true;
    float style_strength_ = 1.0f;
    std::map<EmotionLabel, std::vector<float>> offsets_;
    std::vector<float> style_;
    std::map<std::string, std::int64_t> vocab_;
    std::vector<KokoroVoice> voices_;
    std::size_t current_ = 0;
    bool fixed_ = false;
    bool matched_ = false;
    std::vector<std::int64_t> ids_;
    std::vector<float> dur_;
    std::vector<float> accent_;
};

}  // namespace

std::unique_ptr<ITtsEngine> make_kokoro_engine(const std::string& dir, const Params& params,
                                               const ModelRegistry* registry) {
    return std::make_unique<KokoroEngine>(dir, params, registry);
}

}  // namespace ee
