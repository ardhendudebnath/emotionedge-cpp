#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/prosody/prosody_types.hpp"

namespace ee {

/// One sentence of synthetic demo input and how it is "performed".
struct DemoUtterance {
    std::string text;
    ProsodyTargets delivery;               ///< how the source speaker says it
    std::vector<std::uint16_t> emphasis;   ///< stressed words
};

/// Synthetic source speech plus the matching transcript script, so the whole pipeline can run
/// end to end without microphones or model files. The "speech" is the formant voice (see
/// FormantSynth): the emotion engine hears real prosody in it, while the scripted ASR supplies
/// the words it cannot contain.
struct DemoInput {
    std::vector<float> audio;
    int sample_rate = 16000;
    std::string script_json;  ///< ScriptedAsrEngine script with absolute word timings
    std::vector<std::string> sentences;
};

/// Renders utterances separated by `gap_s` of silence, with `lead_s` / `tail_s` padding.
[[nodiscard]] DemoInput render_demo(const std::vector<DemoUtterance>& utterances, int sample_rate = 16000,
                                    double lead_s = 0.5, double gap_s = 1.2, double tail_s = 1.2);

/// Blueprint p.3 walkthrough: "I can't believe you did this!", said angrily, stress on "believe".
[[nodiscard]] DemoInput make_walkthrough_input(int sample_rate = 16000);

/// The three utterances of the demo conversation: anger, sadness, joy.
[[nodiscard]] std::vector<DemoUtterance> demo_conversation_utterances();

/// Three utterances in three emotions (anger, sadness, joy), for multi-utterance runs.
[[nodiscard]] DemoInput make_demo_conversation(int sample_rate = 16000);

}  // namespace ee
