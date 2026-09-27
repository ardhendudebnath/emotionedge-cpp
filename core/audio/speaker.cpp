#include "core/audio/speaker.hpp"

#include <algorithm>

#include "core/audio/dsp.hpp"
#include "core/audio/pitch.hpp"

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
    rate_ = ctx.pipeline().sample_rate;
    encoder_ = make_speaker_encoder(ctx.params(), ctx.services().models);
    pitch_encoder_ = ctx.params().str("encoder", "pitch") == "pitch";
    max_samples_ = static_cast<std::size_t>(ctx.params().number("max_seconds", 2.0) * rate_);
    audio_.reserve(max_samples_);
}

void SpeakerStage::process(Frame& f) {
    if (f.kind != FrameKind::Audio) return;
    if (f.utterance != utterance_) {
        utterance_ = f.utterance;
        audio_.clear();
    }
    const std::size_t room = max_samples_ - std::min(max_samples_, audio_.size());
    audio_.insert(audio_.end(), f.audio.begin(), f.audio.begin() + static_cast<std::ptrdiff_t>(std::min(room, f.audio.size())));
    if (!f.has(frame_flags::kEndpoint) || audio_.empty()) return;

    const SpeakerEmbedding print = encoder_->embed(audio_, rate_);
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
