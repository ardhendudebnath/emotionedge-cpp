#include "core/runtime/frame.hpp"

#include <array>

namespace ee {

namespace {
constexpr std::array<std::string_view, kFrameKindCount> kKindNames = {
    "none",   "audio",   "transcript", "emotion",  "voiceprint", "utterance",
    "translation", "speech", "synthaudio", "feedback", "playout",    "control"};
}  // namespace

std::string_view to_string(FrameKind kind) noexcept {
    const auto idx = static_cast<std::size_t>(kind);
    return idx < kKindNames.size() ? kKindNames[idx] : "none";
}

std::optional<FrameKind> parse_frame_kind(std::string_view name) noexcept {
    for (std::size_t i = 0; i < kKindNames.size(); ++i) {
        if (kKindNames[i] == name) return static_cast<FrameKind>(i);
    }
    return std::nullopt;
}

void Frame::reset(FrameKind k) noexcept {
    kind = k;
    flags = 0;
    utterance = 0;
    seq = 0;
    stream_pos = 0;
    t_origin = {};
    src_start = 0.0;
    src_end = 0.0;
    sample_rate = 0;
    audio.clear();
    text.clear();
    language.clear();
    detail.clear();
    words.clear();
    stable_words = 0;
    emphasis.clear();
    emotion = {};
    envelope.clear();
    envelope_hop = 0.01f;
    prosody = {};
    style.fill(0.0f);
    voice.fill(0.0f);
    score = 0.0f;
    delta = {};
    axis_confidence = {};
    out_start = 0.0;
    out_end = 0.0;
}

void Frame::copy_header_from(const Frame& other) noexcept {
    utterance = other.utterance;
    seq = other.seq;
    stream_pos = other.stream_pos;
    t_origin = other.t_origin;
    src_start = other.src_start;
    src_end = other.src_end;
}

bool is_droppable(const Frame& frame) noexcept {
    if (frame.is_final()) return false;
    switch (frame.kind) {
    case FrameKind::Transcript:
    case FrameKind::Translation:
    case FrameKind::Emotion: return true;
    default: return false;
    }
}

}  // namespace ee
