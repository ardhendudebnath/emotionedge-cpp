// CTranslate2 + SentencePiece adapter for NLLB-200 (blueprint 3.2). Compiled only with
// -DEE_WITH_CTRANSLATE2=ON.
//
// Model directory: the output of `ct2-transformers-converter --model facebook/nllb-200-distilled-600M
// --quantization int8` plus the model's `sentencepiece.bpe.model` (see ml/export/convert_nllb_ct2.sh).
#include <ctranslate2/devices.h>
#include <ctranslate2/translator.h>
#include <sentencepiece_processor.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <future>
#include <memory>
#include <string>
#include <vector>

#include "core/runtime/batcher.hpp"
#include "core/runtime/log.hpp"
#include "core/runtime/shared_cache.hpp"
#include "core/translate/control_tokens.hpp"
#include "core/translate/draft_gate.hpp"
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

/// Each emphasized source word with the target word that attends to it most. `attention[t][s]`
/// is target token t's weight on source token s (row 0: the language token); `hyp` the target
/// tokens it covers, language token first; the source's words start after `skip` tokens.
std::vector<std::pair<std::uint16_t, std::uint16_t>> align_emphasis(const std::vector<std::vector<float>>& attention,
                                                                    const std::vector<std::string>& tokens,
                                                                    std::size_t skip, const std::vector<std::string>& hyp,
                                                                    const std::vector<std::uint16_t>& emphasis) {
    std::vector<std::pair<std::uint16_t, std::uint16_t>> out;
    const std::vector<int> src_word = piece_words(tokens, skip);
    const std::vector<int> tgt_word = piece_words(hyp, 1);
    for (std::uint16_t s : emphasis) {
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
        if (best_word >= 0) out.push_back({s, static_cast<std::uint16_t>(best_word)});
    }
    return out;
}

/// Where a translation's time went, filled in by the batch that ran it.
struct NllbTrace {
    std::chrono::steady_clock::time_point started;
    double run_ms = 0.0;
    std::size_t batch = 0;
    std::size_t drafts = 0;  ///< caption drafts in that batch
    bool draft = false;
    std::size_t tokens = 0;  ///< target tokens decoded, prefix included
};

/// One translation as the model takes it: source and target-prefix tokens, and the options that
/// must match across a batch.
/// Target tokens any translation may run to (a final's limit).
constexpr std::size_t kMaxDecode = 256;

struct NllbRequest {
    std::vector<std::string> source;
    std::vector<std::string> prefix;
    std::size_t beam = 2;
    bool attention = false;
    std::size_t max_length = kMaxDecode;  ///< target tokens, prefix included
    NllbTrace* trace = nullptr;           ///< the caller's, alive until its batch has run
};

/// A caption draft's decode limit. On FLORES-200 devtest NLLB's Hindi from English is at most
/// 2 x the source pieces + 4; 3 x + 32 leaves room for denser source scripts. A draft past it has
/// gone wrong (a repetition loop) and would hold every session's translation behind it; at
/// worst a long draft's caption is cut short, and the final, which keeps kMaxDecode, is not.
/// (A batch decodes to its longest member's limit: a draft batched with a final gets the final's.)
std::size_t draft_limit(std::size_t prefix_tokens, std::size_t source_pieces) {
    return std::min(kMaxDecode, prefix_tokens + 3 * source_pieces + 32);
}

/// NLLB and its SentencePiece model, loaded once per process for each model, device and setting:
/// every session translates with it. Both are safe to use from several threads. Sessions that
/// translate at the same moment either share batches (`batcher`, one replica: on a GPU a batch
/// costs about what one sentence does) or run on their own replicas (`replicas`, which share
/// the weights).
struct NllbModel {
    sentencepiece::SentencePieceProcessor sp;
    std::unique_ptr<ctranslate2::Translator> translator;
    std::unique_ptr<Batcher<NllbRequest, ctranslate2::TranslationResult>> batcher;
};

std::vector<ctranslate2::TranslationResult> run_batch(ctranslate2::Translator& translator, std::vector<NllbRequest>& batch) {
    std::vector<std::vector<std::string>> sources;
    std::vector<std::vector<std::string>> prefixes;
    ctranslate2::TranslationOptions options;
    options.beam_size = batch.front().beam;  // the batcher groups by beam size
    options.max_decoding_length = 0;
    std::size_t drafts = 0;
    for (NllbRequest& r : batch) {
        sources.push_back(std::move(r.source));
        prefixes.push_back(std::move(r.prefix));
        options.return_attention = options.return_attention || r.attention;
        options.max_decoding_length = std::max(options.max_decoding_length, r.max_length);
        if (r.trace != nullptr && r.trace->draft) ++drafts;
    }
    const auto started = std::chrono::steady_clock::now();
    auto results = translator.translate_batch(sources, prefixes, options);
    const double run_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    for (std::size_t i = 0; i < batch.size(); ++i) {
        NllbTrace* t = batch[i].trace;
        if (t == nullptr) continue;
        const std::size_t tokens = results.at(i).hypotheses.empty() ? 0 : results[i].hypotheses[0].size();
        *t = {started, run_ms, batch.size(), drafts, t->draft, tokens};
    }
    return results;
}

std::shared_ptr<NllbModel> load_nllb(const std::string& model_dir, const std::string& spm, ctranslate2::Device device,
                                     int gpu, const std::string& compute_type, std::size_t threads, std::size_t replicas,
                                     std::size_t max_batch, bool warmup) {
    auto model = std::make_shared<NllbModel>();
    const auto status = model->sp.Load(spm);
    if (!status.ok()) throw ConfigError("cannot load SentencePiece model '" + spm + "': " + status.ToString());
    ctranslate2::models::ModelLoader loader(model_dir);
    loader.device = device;
    loader.device_indices = {gpu};
    loader.num_replicas_per_device = replicas;
    loader.compute_type = ctranslate2::str_to_compute_type(compute_type);
    ctranslate2::ReplicaPoolConfig pool;
    pool.num_threads_per_replica = threads;
    model->translator = std::make_unique<ctranslate2::Translator>(loader, pool);
    if (max_batch > 1) {
        ctranslate2::Translator* translator = model->translator.get();
        model->batcher = std::make_unique<Batcher<NllbRequest, ctranslate2::TranslationResult>>(
            [translator](std::vector<NllbRequest>& batch) { return run_batch(*translator, batch); }, max_batch,
            [](const NllbRequest& r) { return static_cast<int>(r.beam); });
    }
    if (device == ctranslate2::Device::CUDA) {
        log::info("translate: CTranslate2 on cuda (gpu ", gpu, ", ", compute_type, ", ",
                  max_batch > 1 ? "batches of up to " + std::to_string(max_batch) : std::to_string(replicas) + " replica(s)",
                  ")");
    }
    if (warmup) {
        // Each replica's first translation pays for its allocations (and, on a GPU, its cuBLAS
        // setup): not a session's first utterance. One job per replica, all posted at once, so
        // each idle replica takes one.
        std::vector<std::string> tokens{std::string(*nllb_code("en"))};
        std::vector<std::string> pieces;
        model->sp.Encode("Hello, how are you?", &pieces);
        tokens.insert(tokens.end(), pieces.begin(), pieces.end());
        tokens.push_back("</s>");
        const std::vector<std::string> prefix{std::string(*nllb_code("hi"))};
        std::vector<std::future<ctranslate2::TranslationResult>> jobs;
        for (std::size_t i = 0; i < replicas; ++i) {
            auto posted = model->translator->translate_batch_async({tokens}, {prefix});
            jobs.push_back(std::move(posted.at(0)));
        }
        for (auto& job : jobs) (void)job.get();
    }
    return model;
}

class Ct2Translator final : public ITranslator {
public:
    Ct2Translator(const std::string& model_dir, const Params& params)
        : beam_(static_cast<std::size_t>(params.integer("beam", 2))),
          // Whether the model was trained on `<emo=...>` prefixes (the P3 fine-tune). The stage's
          // `control_tokens: on` implies it; with `auto` vanilla NLLB gets plain text.
          control_tokens_(params.flag("model_control_tokens", params.str("control_tokens", "auto") == "on")) {
        const std::string spm = params.str("sentencepiece", model_dir + "/sentencepiece.bpe.model");
        const auto threads = static_cast<std::size_t>(params.integer("threads", 4));
        const auto replicas = static_cast<std::size_t>(std::max<std::int64_t>(1, params.integer("replicas", 1)));
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
        // stream: off | on | auto. A streaming final pass decodes greedily and hands its first
        // words over early (TranslationRequest::stop_after_words). auto: on the CPU, where the
        // final pass is slow enough to matter. Not with batches across sessions.
        const std::string stream = params.str("stream", "off");
        if (stream != "auto" && stream != "on" && stream != "off") throw ConfigError("translate stream must be auto, on or off");
        const ctranslate2::Device device = gpu_visible ? ctranslate2::Device::CUDA : ctranslate2::Device::CPU;
        const std::string compute_type =
            gpu_visible ? params.str("gpu_compute_type", "int8_float16") : params.str("compute_type", "int8");
        const int gpu = static_cast<int>(params.integer("gpu_id", 0));
        // batch: auto | on | off. Sessions translating at once share batches on one replica (auto:
        // on a GPU, where a batch costs about what one sentence does), or each gets a replica
        // (off; on the CPU parallel replicas use more cores).
        const std::string batch = params.str("batch", "auto");
        if (batch != "auto" && batch != "on" && batch != "off") throw ConfigError("translate batch must be auto, on or off");
        const bool batching = replicas > 1 && (batch == "on" || (batch == "auto" && gpu_visible));
        const std::size_t max_batch = batching ? std::max<std::size_t>(replicas, 2) : 1;
        const std::size_t replica_count = batching ? 1 : replicas;
        streams_ = !batching && (stream == "on" || (stream == "auto" && !gpu_visible));
        // busy_drafts: skip | queue. With batches across sessions, a caption draft is skipped
        // while the batch executor is busy, unless this session's last draft is draft_refresh_ms
        // old (queue: drafts always wait their turn, as before).
        const std::string busy_drafts = params.str("busy_drafts", "skip");
        if (busy_drafts != "skip" && busy_drafts != "queue") throw ConfigError("translate busy_drafts must be skip or queue");
        const std::int64_t refresh = params.integer("draft_refresh_ms", 1000);
        if (refresh < 0) throw ConfigError("translate draft_refresh_ms must be >= 0");
        drafts_ = DraftGate(busy_drafts == "skip", std::chrono::milliseconds(refresh));
        static SharedCache<NllbModel> models;
        const std::string key = std::filesystem::weakly_canonical(model_dir).string() + '|' + spm + '|' +
                                (gpu_visible ? "cuda" + std::to_string(gpu) : std::string("cpu")) + '|' + compute_type +
                                '|' + std::to_string(threads) + '|' + std::to_string(replica_count) + '|' +
                                std::to_string(max_batch);
        model_ = models.get(key, [&] {
            return load_nllb(model_dir, spm, device, gpu, compute_type, threads, replica_count, max_batch,
                             params.flag("warmup", true));
        });
    }

    TranslationResult translate(const TranslationRequest& r) override {
        // Under multi-user load, caption drafts give way to finals (DraftGate).
        if (r.draft && model_->batcher) {
            const auto now = DraftGate::Clock::now();
            if (!drafts_.allow(model_->batcher->busy(), now)) {
                log::debug("translate: draft skipped, the batch is busy");
                return {};
            }
            drafts_.ran(now);
        }
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

        const sentencepiece::SentencePieceProcessor& sp = model_->sp;
        std::vector<std::string> control_pieces;
        if (!control.empty()) sp.Encode(std::string(control), &control_pieces);
        std::vector<std::string> pieces;
        sp.Encode(source.plain, &pieces);
        std::vector<std::string> tokens{std::string(*src_code)};
        tokens.insert(tokens.end(), control_pieces.begin(), control_pieces.end());
        tokens.insert(tokens.end(), pieces.begin(), pieces.end());
        tokens.push_back("</s>");

        std::vector<std::string> prefix{std::string(*tgt_code)};
        if (!r.target_prefix.empty()) {
            std::vector<std::string> kept;
            sp.Encode(join_text(r.target_prefix, r.target_language), &kept);
            prefix.insert(prefix.end(), kept.begin(), kept.end());
        }

        TranslationResult out;
        const bool want_attention = r.want_alignment && !source.emphasis.empty();
        if (r.stop_after_words > 0 && streams_) {
            // Greedy, a token at a time: once the word after the ones wanted begins, they are
            // complete, and decoding stops there. The callback runs on CTranslate2's worker,
            // which this thread waits for. It sees every position, the prefix's included.
            std::vector<std::string> target(prefix.begin() + 1, prefix.end());
            std::size_t started = 0;
            ctranslate2::TranslationOptions options;
            options.beam_size = 1;
            options.max_decoding_length = kMaxDecode;
            options.return_attention = want_attention;  // a stopped decode keeps it for what it produced
            options.callback = [&](ctranslate2::GenerationStepResult step) {
                if (step.step < prefix.size()) return false;  // the forced prefix (and language) is reported too
                if (step.token == "</s>") return true;
                if (step.token.rfind("\xE2\x96\x81", 0) == 0 && ++started > r.stop_after_words) {
                    out.partial = true;
                    return true;
                }
                target.push_back(std::move(step.token));
                return false;
            };
            const auto result = model_->translator->translate_batch({tokens}, {prefix}, options);
            if (out.partial) {
                sp.Decode(target, &out.text);
                if (want_attention && !result.at(0).attention.empty()) {
                    std::vector<std::string> hyp{prefix.front()};
                    hyp.insert(hyp.end(), target.begin(), target.end());
                    out.alignment = align_emphasis(result.at(0).attention[0], tokens, 1 + control_pieces.size(), hyp, source.emphasis);
                }
                return out;
            }
            const std::vector<std::string>& hyp = result.at(0).hypotheses.at(0);  // ended first: all of it
            std::vector<std::string> all(hyp.begin() + (hyp.empty() ? 0 : 1), hyp.end());
            std::erase(all, std::string("</s>"));
            sp.Decode(all, &out.text);
            if (want_attention && !result.at(0).attention.empty()) {
                out.alignment = align_emphasis(result.at(0).attention[0], tokens, 1 + control_pieces.size(), hyp, source.emphasis);
            }
            return out;
        }

        NllbTrace trace;
        trace.draft = r.draft;
        const std::size_t limit = r.draft ? draft_limit(prefix.size(), pieces.size()) : kMaxDecode;
        NllbRequest request{tokens, prefix, beam_, want_attention, limit, &trace};
        ctranslate2::TranslationResult result;
        const auto submitted = std::chrono::steady_clock::now();
        if (model_->batcher) {  // with whatever other sessions are translating
            result = model_->batcher->submit(std::move(request));
        } else {
            std::vector<NllbRequest> one{std::move(request)};
            result = std::move(run_batch(*model_->translator, one).at(0));
        }
        log::debug("translate: ", r.draft ? "draft" : "final", " waited ",
                   std::chrono::duration<double, std::milli>(trace.started - submitted).count(), " ms, ran ", trace.run_ms,
                   " ms in a batch of ", trace.batch, " (", trace.drafts, " draft(s)); ", trace.tokens, " of ", limit,
                   " tokens");
        if (trace.tokens + 1 >= limit) {
            log::warn("translate: a ", r.draft ? "draft" : "final", " reached its decode limit (", limit,
                      " tokens): it may be cut short, or the model looped");
        }

        const std::vector<std::string>& hyp = result.hypotheses.at(0);
        std::vector<std::string> target(hyp.begin() + (hyp.empty() ? 0 : 1), hyp.end());  // drop the language token
        std::erase(target, std::string("</s>"));
        sp.Decode(target, &out.text);

        if (want_attention && !result.attention.empty()) {
            out.alignment = align_emphasis(result.attention[0], tokens, 1 + control_pieces.size(), hyp, source.emphasis);
        }
        return out;
    }

    bool understands_control_tokens() const noexcept override { return control_tokens_; }
    bool preserves_markup() const noexcept override { return false; }
    bool reads_markup() const noexcept override { return true; }  // emphasis aligned by attention
    bool streams() const noexcept override { return streams_; }

private:
    std::shared_ptr<NllbModel> model_;  ///< shared with every session on the same model
    std::size_t beam_;
    bool control_tokens_;
    bool streams_ = false;
    DraftGate drafts_;  ///< this session's caption drafts under load
};

}  // namespace

std::unique_ptr<ITranslator> make_ct2_translator(const std::string& model_dir, const Params& params) {
    return std::make_unique<Ct2Translator>(model_dir, params);
}

}  // namespace ee
