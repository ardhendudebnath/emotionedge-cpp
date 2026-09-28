// CTranslate2 + SentencePiece adapter for NLLB-200 (blueprint 3.2). Compiled only with
// -DEE_WITH_CTRANSLATE2=ON.
//
// Model directory: the output of `ct2-transformers-converter --model facebook/nllb-200-distilled-600M
// --quantization int8` plus the model's `sentencepiece.bpe.model` (see ml/export/convert_nllb_ct2.sh).
#include <ctranslate2/devices.h>
#include <ctranslate2/translator.h>
#include <sentencepiece_processor.h>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "core/runtime/log.hpp"
#include "core/translate/control_tokens.hpp"
#include "core/translate/languages.hpp"
#include "core/translate/translator.hpp"

namespace ee {

namespace {

/// Word index of each SentencePiece piece ("▁" starts a new word); -1 for special tokens.
std::vector<int> piece_words(const std::vector<std::string>& pieces, std::size_t skip_front) {
    std::vector<int> out(pieces.size(), -1);
    int word = -1;
    for (std::size_t i = skip_front; i < pieces.size(); ++i) {
        const std::string& p = pieces[i];
        if (p == "</s>") continue;
        if (word < 0 || p.rfind("\xE2\x96\x81", 0) == 0) ++word;  // U+2581 LOWER ONE EIGHTH BLOCK
        out[i] = word;
    }
    return out;
}

class Ct2Translator final : public ITranslator {
public:
    Ct2Translator(const std::string& model_dir, const Params& params)
        : beam_(static_cast<std::size_t>(params.integer("beam", 2))),
          // Whether the model was trained on `<emo=...>` prefixes (the P3 fine-tune). The stage's
          // `control_tokens: on` implies it; with `auto` vanilla NLLB gets plain text.
          control_tokens_(params.flag("model_control_tokens", params.str("control_tokens", "auto") == "on")) {
        const std::string spm = params.str("sentencepiece", model_dir + "/sentencepiece.bpe.model");
        const auto status = sp_.Load(spm);
        if (!status.ok()) throw ConfigError("cannot load SentencePiece model '" + spm + "': " + status.ToString());
        ctranslate2::ReplicaPoolConfig pool;
        pool.num_threads_per_replica = static_cast<std::size_t>(params.integer("threads", 4));
        // device: cpu | cuda | auto. CUDA needs a CTranslate2 built with it and a GPU; without them
        // `auto` quietly, and `cuda` with a warning, run on the CPU, as ORT's providers fall back.
        // The compute type follows the device: compute_type on the CPU, gpu_compute_type on CUDA.
        const std::string requested = params.str("device", "cpu");
        const bool gpu_visible = requested != "cpu" && ctranslate2::get_device_count(ctranslate2::Device::CUDA) > 0;
        if (requested == "cuda" && !gpu_visible) {
            log::warn("translate: no CUDA device for CTranslate2 (not built with CUDA, or no GPU); using the CPU");
        } else if (requested != "cpu" && requested != "cuda" && requested != "auto") {
            throw ConfigError("translate device must be cpu, cuda or auto");
        }
        const ctranslate2::Device device = gpu_visible ? ctranslate2::Device::CUDA : ctranslate2::Device::CPU;
        const std::string compute_type =
            gpu_visible ? params.str("gpu_compute_type", "int8_float16") : params.str("compute_type", "int8");
        const int gpu = static_cast<int>(params.integer("gpu_id", 0));
        translator_ = std::make_unique<ctranslate2::Translator>(model_dir, device,
                                                                ctranslate2::str_to_compute_type(compute_type),
                                                                std::vector<int>{gpu}, false, pool);
        if (gpu_visible) log::info("translate: CTranslate2 on cuda (gpu ", gpu, ", ", compute_type, ")");
        // The first translation pays for allocations (and, on a GPU, cuBLAS setup): not the first
        // utterance.
        if (params.flag("warmup", true)) {
            TranslationRequest warm;
            warm.source = "Hello, how are you?";
            warm.source_language = "en";
            warm.target_language = "hi";
            (void)translate(warm);
        }
    }

    TranslationResult translate(const TranslationRequest& r) override {
        const auto src_code = nllb_code(r.source_language);
        const auto tgt_code = nllb_code(r.target_language);
        if (!src_code || !tgt_code) {
            throw std::runtime_error("no NLLB code for " + std::string(r.source_language) + "->" +
                                     std::string(r.target_language));
        }
        // Vanilla NLLB would translate the control tokens literally: they reach the model only
        // when it is the P3 fine-tune trained on them. The prefix is encoded on its own, so the
        // body's word indices (emphasis) do not count the prefix's words.
        const std::string_view body = strip_control_prefix(r.source);
        std::string_view control = control_tokens_ ? r.source.substr(0, r.source.size() - body.size()) : "";
        while (!control.empty() && control.back() == ' ') control.remove_suffix(1);
        const Markup source = parse_markup(body, r.source_language);

        std::vector<std::string> control_pieces;
        if (!control.empty()) sp_.Encode(std::string(control), &control_pieces);
        std::vector<std::string> pieces;
        sp_.Encode(source.plain, &pieces);
        std::vector<std::string> tokens{std::string(*src_code)};
        tokens.insert(tokens.end(), control_pieces.begin(), control_pieces.end());
        tokens.insert(tokens.end(), pieces.begin(), pieces.end());
        tokens.push_back("</s>");

        std::vector<std::string> prefix{std::string(*tgt_code)};
        if (!r.target_prefix.empty()) {
            std::vector<std::string> kept;
            sp_.Encode(join_text(r.target_prefix, r.target_language), &kept);
            prefix.insert(prefix.end(), kept.begin(), kept.end());
        }

        ctranslate2::TranslationOptions options;
        options.beam_size = beam_;
        options.max_decoding_length = 256;
        options.return_attention = r.want_alignment && !source.emphasis.empty();
        const auto results = translator_->translate_batch({tokens}, {prefix}, options);

        TranslationResult out;
        const std::vector<std::string>& hyp = results.at(0).hypotheses.at(0);
        std::vector<std::string> target(hyp.begin() + (hyp.empty() ? 0 : 1), hyp.end());  // drop the language token
        std::erase(target, std::string("</s>"));
        sp_.Decode(target, &out.text);

        if (options.return_attention && !results[0].attention.empty()) {
            // attention[t][s]: weight of source token s for hypothesis token t (row 0 = language token).
            const auto& attention = results[0].attention[0];
            const std::vector<int> src_word = piece_words(tokens, 1 + control_pieces.size());  // skip lang + prefix
            const std::vector<int> tgt_word = piece_words(hyp, 1);
            for (std::uint16_t s : source.emphasis) {
                int best_word = -1;
                float best = 0.0f;
                std::vector<float> per_word(hyp.size(), 0.0f);
                for (std::size_t t = 0; t < attention.size() && t < hyp.size(); ++t) {
                    if (tgt_word[t] < 0) continue;
                    for (std::size_t k = 0; k < attention[t].size() && k < src_word.size(); ++k) {
                        if (src_word[k] == static_cast<int>(s)) per_word[static_cast<std::size_t>(tgt_word[t])] += attention[t][k];
                    }
                }
                for (std::size_t w = 0; w < per_word.size(); ++w) {
                    if (per_word[w] > best) {
                        best = per_word[w];
                        best_word = static_cast<int>(w);
                    }
                }
                if (best_word >= 0) out.alignment.push_back({s, static_cast<std::uint16_t>(best_word)});
            }
        }
        return out;
    }

    bool understands_control_tokens() const noexcept override { return control_tokens_; }
    bool preserves_markup() const noexcept override { return false; }

private:
    sentencepiece::SentencePieceProcessor sp_;
    std::unique_ptr<ctranslate2::Translator> translator_;
    std::size_t beam_;
    bool control_tokens_;
};

}  // namespace

std::unique_ptr<ITranslator> make_ct2_translator(const std::string& model_dir, const Params& params) {
    return std::make_unique<Ct2Translator>(model_dir, params);
}

}  // namespace ee
