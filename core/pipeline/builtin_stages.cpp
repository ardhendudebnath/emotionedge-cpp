#include "core/pipeline/builtin_stages.hpp"

#include "core/asr/asr_stage.hpp"
#include "core/audio/frontend.hpp"
#include "core/audio/playback.hpp"
#include "core/audio/segmenter.hpp"
#include "core/audio/speaker.hpp"
#include "core/emotion/consistency_stage.hpp"
#include "core/emotion/emotion_stage.hpp"
#include "core/emotion/state_tracker.hpp"
#include "core/pipeline/recorder.hpp"
#include "core/prosody/controller_stage.hpp"
#include "core/translate/translate_stage.hpp"
#include "core/tts/tts_stage.hpp"

namespace ee {

void register_builtin_stages(StageRegistry& r) {
    r.add<FrontendStage>("frontend");
    r.add<SegmenterStage>("segmenter");
    r.add<SpeakerStage>("speaker");
    r.add<StreamingAsrStage>("asr");
    r.add<EmotionEngineStage>("emotion");
    r.add<StateTrackerStage>("state");
    r.add<TranslateStage>("translate");
    r.add<EmotionControllerStage>("controller");
    r.add<TtsStage>("tts");
    r.add<PlaybackStage>("playback");
    r.add<EmotionConsistencyStage>("consistency");
    r.add<RecorderStage>("recorder");
}

StageRegistry builtin_registry() {
    StageRegistry r;
    register_builtin_stages(r);
    return r;
}

}  // namespace ee
