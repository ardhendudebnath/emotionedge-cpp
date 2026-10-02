#include "core/audio/speaker.hpp"

#include <algorithm>
#include <cmath>
#include <string>

#include "core/audio/audio_io.hpp"
#include "core/audio/dsp.hpp"
#include "core/audio/pitch.hpp"
#include "core/runtime/log.hpp"

namespace ee {

SpeakerEmbedding PitchSpeakerEncoder::embed(std::span<const float> audio, int sample_rate) {
    SpeakerEmbedding print{};
    PitchConfig cfg;
    cfg.sample_rate = sample_rate;
    cfg.frame = sample_rate * 40 / 1000;
    cfg.hop = sample_rate * 20 / 1000;  // pitch statistics need no finer resolution
    std::vector<float> f0;
    for (const PitchFrame& pf : track_pitch(audio, cfg)) {
        if (pf.voiced) f0.push_back(pf.f0_hz);
    }
    if (f0.size() >= 5) {
        std::sort(f0.begin(), f0.end());
        const float median = f0[f0.size() / 2];
        const float p10 = f0[f0.size() / 10];
        const float p90 = f0[f0.size() * 9 / 10];
        print[0] = median / 1000.0f;
        print[1] = semitones(p90, p10) / 12.0f;
    }
    print[2] = hf_ratio_db(audio) / 20.0f;
    return print;
}

float voice_print_f0(const SpeakerEmbedding& print) noexcept {
    const float f0 = print[0] * 1000.0f;
    return f0 >= 50.0f && f0 <= 500.0f ? f0 : 0.0f;
}

float voice_similarity(const SpeakerEmbedding& a, const SpeakerEmbedding& b) noexcept {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        dot += static_cast<double>(a[i]) * b[i];
        na += static_cast<double>(a[i]) * a[i];
        nb += static_cast<double>(b[i]) * b[i];
    }
    return na > 0.0 && nb > 0.0 ? static_cast<float>(dot / std::sqrt(na * nb)) : 0.0f;
}

namespace {

bool empty_print(const SpeakerEmbedding& p) noexcept {
    return std::all_of(p.begin(), p.end(), [](float v) { return v == 0.0f; });
}

SpeakerEmbedding normalized(SpeakerEmbedding p) noexcept {
    double norm = 0.0;
    for (float v : p) norm += static_cast<double>(v) * v;
    if (norm <= 0.0) return p;
    const auto scale = static_cast<float>(1.0 / std::sqrt(norm));
    for (float& v : p) v *= scale;
    return p;
}

// a <- normalize((1 - w) a + w normalize(b)): each print counts by its weight, not its length.
void blend(SpeakerEmbedding& a, const SpeakerEmbedding& b, float w) noexcept {
    const SpeakerEmbedding unit = normalized(b);
    for (std::size_t i = 0; i < a.size(); ++i) a[i] = (1.0f - w) * a[i] + w * unit[i];
    a = normalized(a);
}

}  // namespace

// ---- VoiceGate ---------------------------------------------------------------------------------

void VoiceGate::enroll(const SpeakerEmbedding& print) {
    if (empty_print(print)) return;
    if (enrolled_ == 0) {
        reference_ = normalized(print);
        enrolled_ = 1;
        return;
    }
    if (voice_similarity(reference_, print) < cfg_.threshold) return;  // someone else
    enrolled_ = std::min(enrolled_ + 1, cfg_.max_enrolled);
    blend(reference_, print, 1.0f / static_cast<float>(enrolled_));
}

void VoiceGate::hear_output(const SpeakerEmbedding& print) {
    if (empty_print(print)) return;
    if (!have_output_) {
        output_ = normalized(print);
        have_output_ = true;
        return;
    }
    blend(output_, print, 0.5f);  // the TTS may switch voices: follow the latest
}

VoiceGate::Verdict VoiceGate::judge(const SpeakerEmbedding& probe, float* to_speaker, float* to_output) const {
    const float speaker = enrolled_ > 0 ? voice_similarity(reference_, probe) : 0.0f;
    const float output = have_output_ ? voice_similarity(output_, probe) : 0.0f;
    if (to_speaker != nullptr) *to_speaker = speaker;
    if (to_output != nullptr) *to_output = output;
    if (!has_reference() || empty_print(probe)) return Verdict::Unknown;
    if (have_output_ && output >= cfg_.echo_threshold && output > speaker) return Verdict::Echo;
    return speaker >= cfg_.threshold ? Verdict::Speaker : Verdict::Other;
}

std::unique_ptr<ISpeakerEncoder> make_speaker_encoder(const Params& params, const ModelRegistry* registry) {
    const std::string kind = params.str("encoder", "pitch");
    if (kind == "pitch") return std::make_unique<PitchSpeakerEncoder>();
    if (kind == "ecapa") {
#if defined(EE_HAVE_ONNXRUNTIME)
        const std::string path = resolve_model_path(params, registry, "model");
        if (path.empty()) throw ConfigError("speaker encoder 'ecapa' needs a 'model' or 'model_id'");
        return make_ecapa_encoder(path, params, registry);
#else
        (void)registry;
        throw ConfigError("speaker encoder 'ecapa' needs a build with -DEE_WITH_ONNXRUNTIME=ON");
#endif
    }
    (void)registry;
    throw ConfigError("unknown speaker encoder '" + kind + "' (pitch | ecapa)");
}

void SpeakerStage::open(StageContext& ctx) {
    ctx_ = &ctx;
    const Params& p = ctx.params();
    rate_ = ctx.pipeline().sample_rate;
    encoder_ = make_speaker_encoder(p, ctx.services().models);
    pitch_encoder_ = p.str("encoder", "pitch") == "pitch";
    max_samples_ = static_cast<std::size_t>(p.number("max_seconds", 2.0) * rate_);
    audio_.reserve(max_samples_);

    barge_in_ = p.flag("barge_in", false);
    if (barge_in_) {
        // Cosine between pitch statistics says nothing about who is talking.
        if (pitch_encoder_) throw ConfigError("speaker barge_in needs encoder: ecapa");
        VoiceGate::Config gate;
        gate.threshold = p.real("barge_in_threshold", gate.threshold);
        gate.echo_threshold = p.real("echo_threshold", gate.echo_threshold);
        gate.min_enrolled = static_cast<int>(p.integer("barge_in_enrol", gate.min_enrolled));
        if (gate.min_enrolled < 1 || gate.min_enrolled > gate.max_enrolled) {
            throw ConfigError("speaker barge_in_enrol must be 1.." + std::to_string(gate.max_enrolled));
        }
        gate_ = std::make_unique<VoiceGate>(gate);
        probe_samples_ = static_cast<std::size_t>(p.number("barge_in_ms", 1500.0) / 1000.0 * rate_);
        if (probe_samples_ == 0 || probe_samples_ > max_samples_) {
            throw ConfigError("speaker barge_in_ms must be > 0 and within max_seconds");
        }
    }
}

void SpeakerStage::hear_output(const Frame& f) {
    // The translation's own voice, from the first 2 s of each translated utterance (and the
    // TTS's calibration render), so that its echo in the microphone is not taken for a person.
    if (f.utterance != output_utterance_ || f.has(frame_flags::kCalibration)) {
        output_utterance_ = f.utterance;
        output_audio_.clear();
        output_heard_ = false;
    }
    if (output_heard_ || f.sample_rate <= 0) return;
    output_rate_ = f.sample_rate;
    output_audio_.insert(output_audio_.end(), f.audio.begin(), f.audio.end());
    const auto enough = static_cast<std::size_t>(2 * output_rate_);
    if (output_audio_.size() >= enough || f.is_final() || f.has(frame_flags::kCalibration)) {
        gate_->hear_output(encoder_->embed(output_audio_, output_rate_));
        output_heard_ = true;
        output_audio_.clear();
    }
}

void SpeakerStage::decide_barge_in(const Frame& f) {
    probing_ = false;
    float to_speaker = 0.0f;
    float to_output = 0.0f;
    switch (gate_->judge(encoder_->embed(audio_, rate_), &to_speaker, &to_output)) {
    case VoiceGate::Verdict::Unknown: return;
    case VoiceGate::Verdict::Speaker:
        log::debug("speaker: utterance ", f.utterance, " is the speaker talking on (cos ", to_speaker, "): no barge-in");
        return;
    case VoiceGate::Verdict::Echo:
        not_speaker_ = true;
        log::info("speaker: utterance ", f.utterance, " is the translation's own voice (cos ", to_output,
                  "): no barge-in");
        return;
    case VoiceGate::Verdict::Other: break;
    }
    not_speaker_ = true;
    log::info("speaker: another voice talks over the translation (cos ", to_speaker, " to the speaker): barge-in");
    Frame& barge = ctx_->make(FrameKind::Control);
    barge.flags = frame_flags::kBargeIn;
    barge.utterance = f.utterance;
    barge.t_origin = f.t_origin;
    ctx_->emit(barge);
}

void SpeakerStage::process(Frame& f) {
    if (f.kind == FrameKind::SynthAudio) {
        if (barge_in_) hear_output(f);
        return;
    }
    if (f.kind != FrameKind::Audio) return;
    if (f.utterance != utterance_) {
        utterance_ = f.utterance;
        audio_.clear();
        not_speaker_ = false;
        // Speech that starts over the playing translation is judged once barge_in_ms is in.
        const AudioIo* io = ctx_->services().audio;
        probing_ = barge_in_ && gate_->has_reference() && io != nullptr && io->playback_active.load();
    }
    const std::size_t room = max_samples_ - std::min(max_samples_, audio_.size());
    audio_.insert(audio_.end(), f.audio.begin(), f.audio.begin() + static_cast<std::ptrdiff_t>(std::min(room, f.audio.size())));
    if (probing_ && audio_.size() >= probe_samples_) decide_barge_in(f);
    if (!f.has(frame_flags::kEndpoint) || audio_.empty()) return;
    // Shorter than barge_in_ms: judged on what there is (ECAPA needs 0.5 s; less is Unknown).
    if (probing_) decide_barge_in(f);

    const SpeakerEmbedding print = encoder_->embed(audio_, rate_);
    if (barge_in_ && !not_speaker_) gate_->enroll(print);
    Frame& out = ctx_->make(FrameKind::VoicePrint);
    out.copy_header_from(f);
    out.flags = frame_flags::kFinal;
    out.voice = print;
    // The median F0 travels on its own: TTS voices are matched on it whatever the encoder.
    // ECAPA cosine picks the right gender for only 11/24 RAVDESS actors against Kokoro's
    // synthetic Hindi voices; F0 picks it for 23/24.
    out.voice_f0 = voice_print_f0(pitch_encoder_ ? print : PitchSpeakerEncoder().embed(audio_, rate_));
    ctx_->emit(out);
    audio_.clear();
}

}  // namespace ee
