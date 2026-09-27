#include "core/telemetry/telemetry.hpp"

#include <algorithm>
#include <string>

namespace ee::telemetry {

namespace {
constexpr std::uint32_t bit(Milestone m) { return 1u << static_cast<unsigned>(m); }

constexpr std::array<std::string_view, kMilestoneCount> kMilestoneNames = {
    "speech_end", "endpoint",        "asr_final",       "emotion_final", "state_ready",
    "mt_final",   "controller_done", "tts_first_chunk", "first_audio"};

// Blueprint p.2: per-stage targets after the speaker stops talking. Values are budgets to
// validate with this telemetry harness, not measurements.
constexpr std::array<BudgetRow, 8> kBudget = {{
    {"VAD endpoint hangover", bit(Milestone::SpeechEnd), Milestone::Endpoint, 160'000, true},
    {"Emotion fusion", bit(Milestone::Endpoint), Milestone::EmotionFinal, 40'000, false},
    {"ASR final decode", bit(Milestone::Endpoint), Milestone::AsrFinal, 220'000, true},
    {"Emotion state", bit(Milestone::AsrFinal) | bit(Milestone::EmotionFinal), Milestone::StateReady,
     10'000, true},
    {"Translation, final pass", bit(Milestone::StateReady), Milestone::MtFinal, 120'000, true},
    {"Emotion controller", bit(Milestone::MtFinal), Milestone::ControllerDone, 5'000, true},
    {"TTS first chunk", bit(Milestone::ControllerDone), Milestone::TtsFirstChunk, 160'000, true},
    {"Playout buffer", bit(Milestone::TtsFirstChunk), Milestone::FirstAudio, 60'000, true},
}};

std::string row_label(std::string_view name) {
    std::string out;
    for (char c : name) {
        if (c == ' ') out += '_';
        else if (c != ',') out += static_cast<char>(c >= 'A' && c <= 'Z' ? c - 'A' + 'a' : c);
    }
    return out;
}
}  // namespace

std::string_view to_string(Milestone m) noexcept {
    const auto i = static_cast<std::size_t>(m);
    return i < kMilestoneNames.size() ? kMilestoneNames[i] : "unknown";
}

const std::array<BudgetRow, 8>& latency_budget() noexcept { return kBudget; }

Telemetry::Telemetry(std::size_t timeline_slots)
    : epoch_(Clock::now()), slot_count_(std::max<std::size_t>(timeline_slots, 8)),
      slots_(std::make_unique<Slot[]>(slot_count_)) {
    for (std::size_t i = 0; i < kBudget.size(); ++i) {
        rows_[i] = &metrics_.histogram("ee_latency_budget_seconds",
                                       "Per-stage latency after speech end (blueprint p.2 rows)",
                                       {{"row", row_label(kBudget[i].name)}});
    }
    end_to_end_ = &metrics_.histogram("ee_end_to_end_seconds",
                                      "Speech end to first translated audio (target p95 < 800 ms)");
    ecs_ = &metrics_.histogram("ee_emotion_consistency", "Emotion consistency score per clause",
                               {}, 1e-4);
    rtf_ = &metrics_.histogram("ee_asr_real_time_factor", "ASR decode time / audio duration", {},
                               1e-4);
    completed_ = &metrics_.counter("ee_utterances_total", "Utterances that reached playout");
    dropouts_ = &metrics_.counter("ee_audio_dropout_samples_total",
                                  "Samples lost to capture overflow or playout underrun");
    for (std::size_t i = 0; i < slot_count_; ++i) {
        for (auto& t : slots_[i].t) t.store(kUnset, std::memory_order_relaxed);
    }
}

void Telemetry::begin_utterance(std::uint64_t utterance) noexcept {
    Slot& slot = slots_[utterance % slot_count_];
    for (auto& t : slot.t) t.store(kUnset, std::memory_order_relaxed);
    slot.done.store(false, std::memory_order_relaxed);
    slot.utterance.store(utterance, std::memory_order_release);
}

void Telemetry::mark(std::uint64_t utterance, Milestone m, TimePoint t) noexcept {
    if (utterance == 0) return;
    Slot& slot = slots_[utterance % slot_count_];
    if (slot.utterance.load(std::memory_order_acquire) != utterance) return;
    std::int64_t expected = kUnset;
    slot.t[static_cast<std::size_t>(m)].compare_exchange_strong(expected, micros_between(epoch_, t),
                                                               std::memory_order_acq_rel);
    if (m == Milestone::FirstAudio && !slot.done.exchange(true, std::memory_order_acq_rel)) {
        complete(slot);
    }
}

std::int64_t Telemetry::milestone_us(std::uint64_t utterance, Milestone m) const noexcept {
    const Slot& slot = slots_[utterance % slot_count_];
    if (slot.utterance.load(std::memory_order_acquire) != utterance) return -1;
    const std::int64_t v = slot.t[static_cast<std::size_t>(m)].load(std::memory_order_acquire);
    return v == kUnset ? -1 : v;
}

void Telemetry::complete(Slot& slot) noexcept {
    std::array<std::int64_t, kMilestoneCount> t{};
    for (std::size_t i = 0; i < kMilestoneCount; ++i) t[i] = slot.t[i].load(std::memory_order_acquire);

    for (std::size_t r = 0; r < kBudget.size(); ++r) {
        const BudgetRow& row = kBudget[r];
        std::int64_t from = kUnset;
        for (std::size_t i = 0; i < kMilestoneCount; ++i) {
            if ((row.from_mask & (1u << i)) != 0 && t[i] != kUnset) from = std::max(from, t[i]);
        }
        const std::int64_t to = t[static_cast<std::size_t>(row.to)];
        if (from == kUnset || to == kUnset) continue;
        rows_[r]->record(static_cast<std::uint64_t>(std::max<std::int64_t>(0, to - from)));
    }
    const std::int64_t start = t[static_cast<std::size_t>(Milestone::SpeechEnd)];
    const std::int64_t end = t[static_cast<std::size_t>(Milestone::FirstAudio)];
    if (start != kUnset && end != kUnset) {
        end_to_end_->record(static_cast<std::uint64_t>(std::max<std::int64_t>(0, end - start)));
    }
    completed_->inc();
}

void Telemetry::record_ecs(float ecs) noexcept {
    ecs_->record(static_cast<std::uint64_t>(std::clamp(ecs, 0.0f, 1.0f) * 10000.0f + 0.5f));
}

void Telemetry::record_rtf(float rtf) noexcept {
    rtf_->record(static_cast<std::uint64_t>(std::max(rtf, 0.0f) * 10000.0f + 0.5f));
}

void Telemetry::count_dropout(std::uint64_t samples) noexcept { dropouts_->inc(samples); }

std::vector<Telemetry::RowReport> Telemetry::budget_report() const {
    std::vector<RowReport> out;
    out.reserve(kBudget.size());
    for (std::size_t r = 0; r < kBudget.size(); ++r) out.push_back({&kBudget[r], rows_[r]->snapshot()});
    return out;
}

}  // namespace ee::telemetry
