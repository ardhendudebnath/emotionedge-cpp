#include "core/pipeline/demo.hpp"

#include "core/asr/scripted_engine.hpp"
#include "core/tts/formant_synth.hpp"

namespace ee {

namespace {

ProsodyTargets angry_delivery() {
    ProsodyTargets p;
    p.pitch_pct = 18.0f;
    p.range_pct = 60.0f;
    p.rate_pct = 15.0f;
    p.energy_db = 6.0f;
    p.pause_ms = 70.0f;
    p.accent = 1.6f;
    p.final_fall = 0.8f;
    p.hesitation = 0.6f;
    p.tension = 0.85f;
    return p;
}

ProsodyTargets sad_delivery() {
    ProsodyTargets p;
    p.pitch_pct = -12.0f;
    p.range_pct = -35.0f;
    p.rate_pct = -18.0f;
    p.energy_db = -5.0f;
    p.final_fall = -0.1f;
    p.hesitation = 1.5f;
    return p;
}

ProsodyTargets joyful_delivery() {
    ProsodyTargets p;
    p.pitch_pct = 15.0f;
    p.range_pct = 45.0f;
    p.rate_pct = 12.0f;
    p.energy_db = 3.0f;
    p.final_fall = 0.3f;
    p.hesitation = 0.8f;
    return p;
}

}  // namespace

DemoInput render_demo(const std::vector<DemoUtterance>& utterances, int sample_rate, double lead_s, double gap_s,
                      double tail_s) {
    FormantVoice voice;
    voice.sample_rate = sample_rate;
    voice.base_f0_hz = 140.0f;  // the source speaker
    FormantSynth synth(voice);

    DemoInput out;
    out.sample_rate = sample_rate;
    out.audio.assign(static_cast<std::size_t>(lead_s * sample_rate), 0.0f);
    std::vector<Word> all_words;
    SynthesisResult rendered;
    for (std::size_t u = 0; u < utterances.size(); ++u) {
        const DemoUtterance& d = utterances[u];
        SynthesisRequest req;
        req.text = d.text;
        req.language = "en";
        req.prosody = d.delivery;
        req.emphasis = d.emphasis;
        synth.synthesize(req, rendered);
        const double offset = static_cast<double>(out.audio.size()) / sample_rate;
        for (Word w : rendered.words) {
            w.t0 += static_cast<float>(offset);
            w.t1 += static_cast<float>(offset);
            all_words.push_back(w);
        }
        out.audio.insert(out.audio.end(), rendered.audio.begin(), rendered.audio.end());
        const double pad = u + 1 < utterances.size() ? gap_s : tail_s;
        out.audio.insert(out.audio.end(), static_cast<std::size_t>(pad * sample_rate), 0.0f);
        out.sentences.push_back(d.text);
    }
    out.script_json = make_asr_script(all_words, 0.0, "en");
    return out;
}

DemoInput make_walkthrough_input(int sample_rate) {
    return render_demo({{"I can't believe you did this!", angry_delivery(), {2}}}, sample_rate);
}

std::vector<DemoUtterance> demo_conversation_utterances() {
    return {{"I can't believe you did this!", angry_delivery(), {2}},
            {"I'm so sorry, I didn't mean to hurt you.", sad_delivery(), {}},
            {"This is amazing, thank you so much!", joyful_delivery(), {2}}};
}

DemoInput make_demo_conversation(int sample_rate) { return render_demo(demo_conversation_utterances(), sample_rate); }

}  // namespace ee
