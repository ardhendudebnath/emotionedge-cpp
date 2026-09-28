#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "core/asr/transcript.hpp"
#include "core/emotion/emotion_types.hpp"
#include "core/prosody/prosody_types.hpp"
#include "core/runtime/clock.hpp"

namespace ee {

/// What a Frame carries. Stages dispatch on it and graph edges can filter on it.
enum class FrameKind : std::uint8_t {
    None,
    Audio,        ///< 16 kHz mono speech audio from the segmenter (1.3), or an endpoint marker
    Transcript,   ///< ASR hypothesis, partial or final (2.1)
    Emotion,      ///< fused emotion estimate (2.2)
    VoicePrint,   ///< speaker embedding (1.4)
    Utterance,    ///< final transcript joined with emotion and emphasis (3.1)
    Translation,  ///< target-language text, draft or final (3.2)
    Speech,       ///< synthesis request: text + prosody plan + style vector (4.1)
    SynthAudio,   ///< synthesized audio chunk (4.2)
    Feedback,     ///< emotion-consistency score and ΔVAD, 5.2 -> 4.1
    Playout,      ///< where an utterance landed on the output timeline (5.1)
    Control,      ///< end-of-stream, barge-in
};
inline constexpr std::size_t kFrameKindCount = 12;

[[nodiscard]] std::string_view to_string(FrameKind kind) noexcept;
[[nodiscard]] std::optional<FrameKind> parse_frame_kind(std::string_view name) noexcept;
[[nodiscard]] constexpr std::uint32_t kind_bit(FrameKind kind) noexcept {
    return 1u << static_cast<unsigned>(kind);
}
inline constexpr std::uint32_t kAllKinds = 0xFFFFFFFFu;

/// Bits in Frame::flags.
namespace frame_flags {
inline constexpr std::uint32_t kFinal = 1u << 0;        ///< final hypothesis / last chunk of an utterance
inline constexpr std::uint32_t kSpeechStart = 1u << 1;  ///< first audio frame of an utterance
inline constexpr std::uint32_t kEndpoint = 1u << 2;     ///< speech ended (after the hangover)
inline constexpr std::uint32_t kBargeIn = 1u << 3;      ///< the user started talking over playback
inline constexpr std::uint32_t kEndOfStream = 1u << 4;  ///< nothing follows on this edge
inline constexpr std::uint32_t kClauseEnd = 1u << 5;    ///< last audio chunk of a TTS clause
inline constexpr std::uint32_t kCalibration = 1u << 6;  ///< neutral reference render, not for playout
}  // namespace frame_flags

/// The unit of data exchanged between stages (blueprint p.4: `IStage::process(Frame&)`).
///
/// A Frame is a flat record rather than a variant so queue slots can be reused in place:
/// copy-assigning into a slot reuses the slot's existing buffer capacity, which keeps
/// steady-state traffic allocation-free. `kind` says which payload fields are meaningful.
struct Frame {
    // ---- header ----------------------------------------------------------------------------
    FrameKind kind = FrameKind::None;
    std::uint32_t flags = 0;
    std::uint64_t utterance = 0;  ///< utterance id from the segmenter; 0 = not tied to one
    std::uint32_t seq = 0;        ///< chunk or clause index within the utterance
    std::int64_t stream_pos = 0;  ///< input-timeline sample index (16 kHz) of the first sample
    TimePoint t_origin{};         ///< wall-clock time of the source audio: the latency reference
    double src_start = 0.0;       ///< utterance start on the input timeline, seconds
    double src_end = 0.0;         ///< utterance speech end on the input timeline, seconds

    // ---- payload ---------------------------------------------------------------------------
    int sample_rate = 0;
    std::vector<float> audio;              ///< Audio, SynthAudio
    std::string text;                      ///< transcript, translation, or TTS clause text
    std::string language;                  ///< language of `text` (ISO 639-1: "en", "hi", ...)
    std::string detail;                    ///< Translation: the exact MT input (control tokens + markup)
    std::vector<Word> words;               ///< Transcript, Utterance: word timings
    std::uint32_t stable_words = 0;        ///< Transcript: words committed by LocalAgreement
    std::vector<std::uint16_t> emphasis;   ///< indices of emphasized words
    EmotionState emotion;                  ///< the source emotion, carried down to 5.2
    std::vector<float> envelope;           ///< Emotion: energy envelope (dB) every envelope_hop s
    float envelope_hop = 0.01f;
    ProsodyTargets prosody;                ///< Speech, SynthAudio: controller plan
    StyleVector style{};                   ///< Speech: style vector for the TTS
    SpeakerEmbedding voice{};              ///< VoicePrint: speaker embedding
    float voice_f0 = 0.0f;                 ///< VoicePrint: speaker's median F0 in Hz (0 = unknown)
    float score = 0.0f;                    ///< Feedback: emotion consistency score
    Vad delta;                             ///< Feedback: VAD_out - VAD_src
    Vad axis_confidence;                   ///< Emotion, Feedback: per-axis confidence of `emotion`
    double out_start = 0.0;                ///< Playout: output-timeline seconds
    double out_end = 0.0;

    [[nodiscard]] bool has(std::uint32_t f) const noexcept { return (flags & f) != 0; }
    [[nodiscard]] bool is_final() const noexcept { return has(frame_flags::kFinal); }
    [[nodiscard]] bool is_end_of_stream() const noexcept {
        return kind == FrameKind::Control && has(frame_flags::kEndOfStream);
    }
    [[nodiscard]] double duration() const noexcept {
        return sample_rate > 0 ? static_cast<double>(audio.size()) / sample_rate : 0.0;
    }

    /// Clears every field for reuse as a `k` frame while keeping allocated capacity.
    void reset(FrameKind k) noexcept;
    /// Copies the header (utterance, timing, origin) but none of the payload.
    void copy_header_from(const Frame& other) noexcept;
};

/// Frames that may be dropped or superseded under backpressure: non-final hypotheses.
/// Audio is never droppable ("drops stale partials, never audio").
[[nodiscard]] bool is_droppable(const Frame& frame) noexcept;

}  // namespace ee
