#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/asr/asr_engine.hpp"
#include "core/asr/local_agreement.hpp"
#include "core/runtime/model_registry.hpp"
#include "core/runtime/stage.hpp"

namespace ee {

/// Stage 2.1 "Streaming ASR". Buffers each utterance's audio, re-decodes it every
/// `partial_interval_ms` while the speaker talks, turns the hypotheses into stable partials
/// with LocalAgreement-2, and at the endpoint emits the final transcript with word timestamps
/// and the detected language. Records the real-time factor of every decode.
class StreamingAsrStage final : public IStage {
public:
    void open(StageContext& ctx) override;
    void process(Frame& frame) override;

private:
    void decode(bool final);
    void emit(const std::vector<Word>& words, std::uint32_t stable, bool final);

    StageContext* ctx_ = nullptr;
    HotSwap<IAsrEngine> engine_;
    LocalAgreement agreement_;
    int rate_ = 16000;
    std::size_t partial_interval_ = 8000;
    std::size_t min_partial_ = 16000;
    std::string language_hint_;
    std::string language_;
    std::vector<float> audio_;
    std::size_t next_partial_at_ = 0;
    std::uint64_t utterance_ = 0;
    std::int64_t start_pos_ = 0;
    double src_start_ = 0.0;
    double src_end_ = 0.0;
    TimePoint origin_{};
    std::string last_partial_;
};

}  // namespace ee
