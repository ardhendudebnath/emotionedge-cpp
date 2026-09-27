// Piper / VITS voices through ONNX Runtime, phonemized with espeak-ng (blueprint 4.2 "fast
// fallback: Piper / VITS"). Compiled only with -DEE_WITH_PIPER=ON.
//
// A voice is `<name>.onnx` plus `<name>.onnx.json` (sample rate, espeak voice, phoneme_id_map,
// inference scales). Piper voices take rate as length_scale; energy and emphasis pre-pauses are
// applied around the model. Pitch mean/range and the style vector need StyleTTS2 (phase 3).
#include <espeak-ng/speak_lib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "core/audio/dsp.hpp"
#include "core/runtime/onnx.hpp"
#include "core/translate/languages.hpp"
#include "core/tts/tts_engine.hpp"

namespace ee {

namespace {

std::mutex& espeak_mutex() {
    static std::mutex m;  // espeak-ng keeps global state
    return m;
}

std::vector<char32_t> codepoints(std::string_view s) {
    std::vector<char32_t> out;
    for (std::size_t i = 0; i < s.size();) {
        const auto c = static_cast<unsigned char>(s[i]);
        int n = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
        char32_t cp = n == 1 ? c : n == 2 ? (c & 0x1Fu) : n == 3 ? (c & 0x0Fu) : (c & 0x07u);
        for (int k = 1; k < n && i + static_cast<std::size_t>(k) < s.size(); ++k) {
            cp = (cp << 6) | (static_cast<unsigned char>(s[i + static_cast<std::size_t>(k)]) & 0x3Fu);
        }
        out.push_back(cp);
        i += static_cast<std::size_t>(n);
    }
    return out;
}

std::string utf8(char32_t cp) {
    std::string out;
    if (cp < 0x80) {
        out += static_cast<char>(cp);
    } else if (cp < 0x800) {
        out += static_cast<char>(0xC0 | (cp >> 6));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        out += static_cast<char>(0xE0 | (cp >> 12));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        out += static_cast<char>(0xF0 | (cp >> 18));
        out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    }
    return out;
}

class PiperEngine final : public ITtsEngine {
public:
    PiperEngine(const std::string& model_path, const Params& params, const ModelRegistry* registry)
        : session_(onnx::load_session(model_path, onnx::session_config(params, registry))) {
        const std::string config_path = params.str("voice_config", model_path + ".json");
        std::ifstream in(config_path);
        if (!in) throw ConfigError("Piper voice config not found: " + config_path);
        nlohmann::json cfg;
        in >> cfg;
        sample_rate_ = cfg.at("audio").at("sample_rate").get<int>();
        voice_ = cfg.at("espeak").at("voice").get<std::string>();
        if (const auto inf = cfg.find("inference"); inf != cfg.end()) {
            noise_scale_ = inf->value("noise_scale", noise_scale_);
            length_scale_ = inf->value("length_scale", length_scale_);
            noise_w_ = inf->value("noise_w", noise_w_);
        }
        multi_speaker_ = cfg.value("num_speakers", 1) > 1;
        speaker_ = params.integer("speaker", 0);
        for (const auto& [phoneme, ids] : cfg.at("phoneme_id_map").items()) {
            ids_[phoneme] = ids.get<std::vector<std::int64_t>>();
        }

        std::lock_guard lock(espeak_mutex());
        const std::string data = params.str("espeak_data");
        if (espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 0, data.empty() ? nullptr : data.c_str(), 0) < 0) {
            throw ConfigError("cannot initialize espeak-ng (set espeak_data to its data directory)");
        }
    }

    int sample_rate() const noexcept override { return sample_rate_; }

    void synthesize(const SynthesisRequest& req, SynthesisResult& out) override {
        out.sample_rate = sample_rate_;
        out.audio.clear();
        out.words.clear();
        const ProsodyTargets& p = req.prosody;
        const float rate = std::clamp(1.0f + p.rate_pct / 100.0f, 0.5f, 2.0f);
        const float gain = db_to_gain(p.energy_db);

        // Split before each emphasized word so it can get its pre-pause and a louder, slower read.
        const std::vector<std::string> words = split_words(req.text, req.language);
        std::vector<std::pair<std::string, bool>> segments;
        for (std::size_t i = 0; i < words.size(); ++i) {
            const bool emph = std::find(req.emphasis.begin(), req.emphasis.end(), static_cast<std::uint16_t>(i)) != req.emphasis.end();
            if (segments.empty() || emph || segments.back().second) {
                segments.push_back({words[i], emph});
            } else {
                segments.back().first += " " + words[i];
            }
        }
        for (const auto& [text, emph] : segments) {
            if (emph && p.pause_ms > 0.0f) out.audio.insert(out.audio.end(), static_cast<std::size_t>(p.pause_ms / 1000.0f * sample_rate_), 0.0f);
            const std::vector<std::int64_t> ids = phoneme_ids(text);
            if (ids.size() < 3) continue;
            const float length = length_scale_ / rate * (emph ? 1.1f : 1.0f);
            run(ids, length, emph ? gain * db_to_gain(3.0f) : gain, out.audio);
        }
    }

private:
    std::vector<std::int64_t> phoneme_ids(const std::string& text) {
        std::string phonemes;
        {
            std::lock_guard lock(espeak_mutex());
            espeak_SetVoiceByName(voice_.c_str());
            const void* cursor = text.c_str();
            while (cursor != nullptr) {
                const char* clause = espeak_TextToPhonemes(&cursor, espeakCHARS_UTF8, espeakPHONEMES_IPA);
                if (clause == nullptr) break;
                if (!phonemes.empty()) phonemes += ' ';
                phonemes += clause;
            }
        }
        // Clause terminators are lost by espeak_TextToPhonemes; restore the sentence's final mark.
        const char last = text.empty() ? '.' : text.back();
        phonemes += (last == '?' || last == '!' || last == ',') ? last : '.';

        // Piper's mapping: BOS, then each phoneme followed by PAD, then EOS.
        std::vector<std::int64_t> ids;
        const auto append = [&](const std::string& key) {
            const auto it = ids_.find(key);
            if (it == ids_.end()) return false;
            ids.insert(ids.end(), it->second.begin(), it->second.end());
            return true;
        };
        append("^");
        append("_");
        for (char32_t cp : codepoints(phonemes)) {
            if (append(utf8(cp))) append("_");
        }
        append("$");
        return ids;
    }

    void run(const std::vector<std::int64_t>& ids, float length_scale, float gain, std::vector<float>& audio) {
        std::vector<std::int64_t> input = ids;
        std::array<std::int64_t, 1> lengths{static_cast<std::int64_t>(input.size())};
        std::array<float, 3> scales{noise_scale_, length_scale, noise_w_};
        std::array<std::int64_t, 1> sid{speaker_};
        const std::array<std::int64_t, 2> input_dims{1, static_cast<std::int64_t>(input.size())};
        const std::array<std::int64_t, 1> one{1};
        const std::array<std::int64_t, 1> three{3};

        std::vector<Ort::Value> tensors;
        tensors.push_back(Ort::Value::CreateTensor<std::int64_t>(memory_, input.data(), input.size(), input_dims.data(), 2));
        tensors.push_back(Ort::Value::CreateTensor<std::int64_t>(memory_, lengths.data(), 1, one.data(), 1));
        tensors.push_back(Ort::Value::CreateTensor<float>(memory_, scales.data(), 3, three.data(), 1));
        std::vector<const char*> names{"input", "input_lengths", "scales"};
        if (multi_speaker_) {
            tensors.push_back(Ort::Value::CreateTensor<std::int64_t>(memory_, sid.data(), 1, one.data(), 1));
            names.push_back("sid");
        }
        const char* output_names[] = {"output"};
        auto result = session_.Run(Ort::RunOptions{nullptr}, names.data(), tensors.data(), tensors.size(), output_names, 1);
        const auto info = result[0].GetTensorTypeAndShapeInfo();
        const float* samples = result[0].GetTensorData<float>();
        const std::size_t count = info.GetElementCount();
        audio.reserve(audio.size() + count);
        for (std::size_t i = 0; i < count; ++i) audio.push_back(soft_limit(samples[i] * gain, 0.95f));
    }

    Ort::Session session_;
    Ort::MemoryInfo memory_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    int sample_rate_ = 22050;
    std::string voice_;
    float noise_scale_ = 0.667f;
    float length_scale_ = 1.0f;
    float noise_w_ = 0.8f;
    bool multi_speaker_ = false;
    std::int64_t speaker_ = 0;
    std::map<std::string, std::vector<std::int64_t>> ids_;
};

}  // namespace

std::unique_ptr<ITtsEngine> make_piper_engine(const std::string& model_path, const Params& params,
                                              const ModelRegistry* registry) {
    return std::make_unique<PiperEngine>(model_path, params, registry);
}

}  // namespace ee
