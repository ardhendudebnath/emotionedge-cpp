// Text emotion classifier (DistilRoBERTa, blueprint 2.2 "LEXICAL") through ONNX Runtime.
// Compiled only with -DEE_WITH_ONNXRUNTIME=ON.
//
// Model directory (produced by ml/export/export_lexical_onnx.py):
//   model.onnx      inputs "input_ids", "attention_mask" int64 [1, tokens]; output "logits" [1, classes]
//   tokenizer.json  the Hugging Face byte-level BPE tokenizer
//   labels.json     class -> V·A·D map (ClassEmotionMap), plus "languages" and "max_tokens"
#include <nlohmann/json.hpp>

#include <array>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "core/emotion/bpe_tokenizer.hpp"
#include "core/emotion/class_mapping.hpp"
#include "core/emotion/lexical.hpp"
#include "core/runtime/onnx.hpp"

namespace ee {

namespace {

namespace fs = std::filesystem;

// Whisper and typed text use curly quotes; the model was trained mostly on ASCII ones.
std::string normalize_quotes(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    for (std::size_t i = 0; i < text.size(); ++i) {
        if (i + 2 < text.size() && static_cast<unsigned char>(text[i]) == 0xE2 &&
            static_cast<unsigned char>(text[i + 1]) == 0x80) {
            const auto c = static_cast<unsigned char>(text[i + 2]);
            if (c == 0x98 || c == 0x99) {  // ‘ ’
                out += '\'';
                i += 2;
                continue;
            }
            if (c == 0x9C || c == 0x9D) {  // “ ”
                out += '"';
                i += 2;
                continue;
            }
        }
        out += text[i];
    }
    return out;
}

class OnnxLexicalEmotionModel final : public ILexicalEmotionModel {
public:
    OnnxLexicalEmotionModel(const fs::path& dir, const onnx::SessionConfig& config)
        : session_(onnx::load_session((dir / "model.onnx").string(), config)),
          tokenizer_(ByteLevelBpeTokenizer::from_tokenizer_json(dir / "tokenizer.json")),
          map_(ClassEmotionMap::from_file(dir / "labels.json")) {
        std::ifstream in(dir / "labels.json", std::ios::binary);
        const nlohmann::json j = nlohmann::json::parse(in);
        languages_ = j.value("languages", std::vector<std::string>{"en"});
        max_tokens_ = j.value("max_tokens", std::size_t{128});
        const std::size_t outputs = session_.GetOutputCount();
        if (outputs == 0) throw ConfigError("lexical model '" + dir.string() + "' has no outputs");
        output_ = session_.GetOutputNameAllocated(0, allocator_).get();
    }

    bool supports(std::string_view language) const override {
        for (const std::string& l : languages_) {
            if (l == "*" || l == language) return true;
        }
        return false;
    }

    ModalityEstimate estimate(std::string_view text, std::string_view language) override {
        if (!supports(language)) return {};
        std::string clean = normalize_quotes(text);
        if (clean.find_first_not_of(" \t\r\n") == std::string::npos) return {};
        // Partial transcripts repeat: the same text gets the same answer without a model run.
        if (clean == last_text_) return last_;

        ids_ = tokenizer_.encode(clean, max_tokens_);
        mask_.assign(ids_.size(), 1);
        const std::array<std::int64_t, 2> dims{1, static_cast<std::int64_t>(ids_.size())};
        std::array<Ort::Value, 2> inputs{
            Ort::Value::CreateTensor<std::int64_t>(memory_, ids_.data(), ids_.size(), dims.data(), dims.size()),
            Ort::Value::CreateTensor<std::int64_t>(memory_, mask_.data(), mask_.size(), dims.data(), dims.size())};
        const char* input_names[] = {"input_ids", "attention_mask"};
        const char* output_names[] = {output_.c_str()};
        auto result = session_.Run(Ort::RunOptions{nullptr}, input_names, inputs.data(), inputs.size(), output_names, 1);
        const auto info = result[0].GetTensorTypeAndShapeInfo();
        const float* logits = result[0].GetTensorData<float>();
        last_ = map_.from_logits(std::span<const float>(logits, info.GetElementCount()));
        last_text_ = std::move(clean);
        return last_;
    }

private:
    Ort::Session session_;
    Ort::AllocatorWithDefaultOptions allocator_;
    Ort::MemoryInfo memory_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    ByteLevelBpeTokenizer tokenizer_;
    ClassEmotionMap map_;
    std::vector<std::string> languages_;
    std::size_t max_tokens_ = 128;
    std::string output_;
    std::vector<std::int64_t> ids_;
    std::vector<std::int64_t> mask_;
    std::string last_text_;
    ModalityEstimate last_;
};

}  // namespace

std::unique_ptr<ILexicalEmotionModel> make_onnx_lexical_model(const std::string& dir, const Params& params,
                                                              const ModelRegistry* registry) {
    return std::make_unique<OnnxLexicalEmotionModel>(dir, onnx::session_config(params, registry));
}

}  // namespace ee
