#pragma once

#include "core/runtime/stage_registry.hpp"

namespace ee {

/// Registers every built-in stage type under the names the pipeline YAML uses:
///
///   frontend (1.1/1.2)   segmenter (1.3)   speaker (1.4)   asr (2.1)   emotion (2.2)
///   state (3.1)          translate (3.2)   controller (4.1) tts (4.2)
///   playback (5.1)       consistency (5.2) recorder (5.3)
///
/// Registration is explicit (not static initializers) so linking the static libraries never
/// silently drops a stage.
void register_builtin_stages(StageRegistry& registry);

/// A registry with all built-in stages.
[[nodiscard]] StageRegistry builtin_registry();

}  // namespace ee
