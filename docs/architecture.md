# Architecture: blueprint → code

This maps each box of the *EmotionEdge-C++ Architecture* blueprint (v2) to the code that
implements it, with its status in this first pass.

**Status key:**
- **done**: implemented and tested.
- **stand-in**: a transparent built-in engine that keeps the pipeline runnable without model files.
- **adapter**: the real engine's adapter, compiled with its CMake option (off by default).
- **planned**: named roadmap phase.

## Real-time C++ core

One process, no Python at runtime. Stages exchange `Frame`s (`core/runtime/frame.hpp`) through
lock-free SPSC queues and implement the blueprint's contract:

```cpp
struct IStage { virtual void process(Frame&) = 0; };   // core/runtime/stage.hpp (+ open/tick/close)
```

The graph (stages, edges, threads) is YAML: `config/pipeline.yaml`, or `config/pipeline.engines.yaml`
for the real engines.

### 01 · Capture

| Box | Code | Status |
|---|---|---|
| 1.1 Audio capture: miniaudio, 16 kHz mono f32, 20 ms frames, zero-alloc callback, SPSC ring | `audio/device.cpp`, `audio/ring_buffer.hpp`, `audio/frontend.cpp` | done; miniaudio **adapter** (`EE_WITH_MINIAUDIO`) |
| 1.2 Front-end DSP: echo cancel (reference = our TTS), neural noise suppression, AGC + loudness | `audio/dsp.cpp`, `audio/frontend.cpp` | AGC + limiter **done**; the playback → AEC reference path is **done**; RNNoise / WebRTC AEC3 plug into `INoiseSuppressor` / `IEchoCanceller` (**planned**) |
| 1.3 VAD & segmenter: speech/silence every ~30 ms, 160 ms hangover, partial chunks, barge-in | `audio/speech_detector.cpp`, `audio/silero_detector.cpp`, `audio/segmenter.cpp` | **done** (energy detector **stand-in**; Silero v5/v6 **adapter**, `EE_WITH_ONNXRUNTIME`) |
| 1.4 Speaker encoder: ECAPA-TDNN, 192-d voice print for the TTS | `audio/speaker.cpp`, `audio/ecapa_encoder.cpp` | ECAPA-TDNN (SpeechBrain, exact ONNX export) **done**; pitch voice print **stand-in**. Every print also carries the speaker's median F0, which picks the TTS voice. ECAPA cosine against synthetic voices gets the gender right for only 11/24 RAVDESS actors; F0 gets 23/24 |

### 02 · Perceive

| Box | Code | Status |
|---|---|---|
| 2.1 Streaming ASR: whisper.cpp, LocalAgreement-2, partial + final, word timestamps, language ID | `asr/local_agreement.cpp`, `asr/asr_stage.cpp`, `asr/whisper_engine.cpp` | **done**; scripted **stand-in**; whisper.cpp **adapter** (`EE_WITH_WHISPER`) |
| 2.2 Emotion engine: acoustic (emotion2vec), prosody (F0 · energy · rate · jitter), lexical (DistilRoBERTa), gated late fusion, EMA | `emotion/prosody_features.cpp`, `emotion/acoustic.cpp`, `emotion/onnx_acoustic.cpp`, `emotion/lexical.cpp`, `emotion/fusion.cpp`, `emotion/emotion_stage.cpp` | fusion, EMA, prosody features (YIN F0) **done**; emotion2vec+ base and DistilRoBERTa (with a C++ byte-level BPE tokenizer) through ONNX Runtime **done** (phase 2: classifiers mapped to V·A·D by `emotion/class_mapping.cpp`); prosody-rules and lexicon remain as **stand-ins** for builds without ORT |

### 03 · Understand

| Box | Code | Status |
|---|---|---|
| 3.1 Context & emotion state: join text + emotion, hysteresis, emphasis = energy peaks on word timestamps | `emotion/state_tracker.cpp`, `emotion/fusion.cpp` | **done** |
| 3.2 Emotion-aware translation: NLLB-200 INT8 on CTranslate2, `<emo=… a=… reg=…>` control tokens, wait-k drafts, final re-translation keeping the prefix, emphasis → target words, glossary, neutral fallback below τ | `translate/*` | **done**; phrasebook **stand-in**; CTranslate2 + SentencePiece **adapter** (`EE_WITH_CTRANSLATE2`); phase-2 NLLB-600M LoRA trained on the control prefix (`ml/train/finetune_nllb_lora.py`). MT output is stripped of any echoed `<emo=…>` span before TTS and captions |
| 3.3 Expressivity profiles per language | `prosody/expressivity.cpp`, `config/expressivity.yaml` | **done** (values are placeholders to calibrate) |

### 04 · Express

| Box | Code | Status |
|---|---|---|
| 4.1 Emotion controller: V·A·D → pitch, range, rate, energy, pauses, voice quality; relative to the target baseline; emphasis boosts; closed-loop correction; 128-d style vector | `prosody/controller.cpp`, `prosody/controller_stage.cpp`, `prosody/style.cpp` | rules v1 + closed loop **done**. For Kokoro, per-emotion offsets in its 256-d style space are learned against an emotion classifier (`ml/train/learn_style_offsets.py`) and applied by V·A·D strength and confidence. The 128-d placeholder anchors remain for other engines |
| 4.2 Expressive TTS: StyleTTS2 on style vector + voice print, clause chunker, HiFi-GAN 24 kHz, Piper/VITS fallback | `tts/clause_chunker.cpp`, `tts/formant_synth.cpp`, `tts/piper_engine.cpp`, `tts/tts_stage.cpp` | chunker **done** (first clause cappable for slow vocoders). Kokoro-82M (StyleTTS2 family, 24 kHz) **done** (`tts/kokoro_engine.cpp`, `tts/kokoro_g2p.cpp` = misaki's G2P ported to C++): the controller drives its durations, F0 shift, range, accents and final contour. Piper **fallback**; formant **stand-in** |

### 05 · Deliver

| Box | Code | Status |
|---|---|---|
| 5.1 Playback: 60 ms jitter buffer, chunk crossfade, AEC reference, source ducking | `audio/playback.cpp`, `pipeline/session.cpp` (`mix_with_ducking`) | **done** |
| 5.2 Emotion consistency: ECS = 1 − ‖ΔVAD‖/2√3, nudge 4.1 below threshold, log per utterance | `emotion/consistency_stage.cpp`, `telemetry/ecs.hpp` | **done**; with the engines config the output is judged by emotion2vec+ (language-agnostic, one session shared with 2.2) |
| 5.3 Outputs: speaker, live captions with emotion tags, gRPC/WebSocket, CLI/desktop, WAV + SRT/JSON | `pipeline/recorder.cpp`, `apps/cli` | CLI, WAV, SRT, JSON **done**; gRPC/WebSocket and desktop **planned** (phase 5) |

## Edge runtime (cross-cutting)

| Service | Code | Status |
|---|---|---|
| Inference runtime: ORT execution providers CPU / CUDA / TensorRT / OpenVINO / CoreML / NNAPI, IOBinding, INT8/FP16 | `runtime/onnx.cpp`; device profiles in `models/manifest.json` | **adapter** (EP selection; Silero binds its output buffers) |
| Scheduler: thread per stage, core-pinned, lock-free queues + backpressure, drops stale partials, never audio | `runtime/graph.cpp`, `runtime/spsc_queue.hpp`, `runtime/thread_util.cpp` | **done** (deterministic single-thread mode for offline runs and tests) |
| Telemetry: per-stage p50/p95, RTF, ECS, queue depth, Perfetto trace, Prometheus | `telemetry/*` | **done** |
| Model registry: SHA-256 verified files, hot-swap per language pair, device profiles | `runtime/model_registry.cpp`, `runtime/sha256.cpp` | **done** |
| Config & plugins: YAML pipeline graph, `IStage` for new engines, language packs | `runtime/config.cpp`, `runtime/stage_registry.cpp`, `pipeline/builtin_stages.cpp` | **done** |

## Threading model (p.2)

| Thread | Blueprint | `config/pipeline.yaml` |
|---|---|---|
| T0 | audio I/O callback | miniaudio device callbacks (`audio/device.cpp`) or `PacedFeeder`/`PacedDrain` in benchmarks |
| T1 | DSP + VAD, 20 ms tick | `frontend`, `segmenter` |
| T2 | streaming ASR | `asr` |
| T3 | emotion engine | `emotion`, `speaker`, `consistency` |
| T4 | state + MT + controller | `state`, `translate`, `controller` |
| T5 | TTS + vocoder | `tts` |
| T6 | playback | `playback`, `recorder` |
| T7 | telemetry flush | built into the scheduler (`telemetry.flush_ms`) |

A thread pins to a core set (`cores: [2, 3, 4, 5]`, or `core: N`). Worker threads that an engine
starts from it inherit the set: whisper.cpp starts its ggml workers on every decode. So the set
must cover the stage's `threads`, and `validate()` rejects a config where it does not. With four
spinning whisper workers pinned to one core, the real-time pipeline stalled.

## Latency budget (p.2)

`ee_bench` measures every row from per-utterance milestones (`telemetry/telemetry.cpp`). The
table below is the real-time paced run over 9 utterances (Release build on a WSL2 x86-64 dev
machine, stand-in engines). It measures the runtime's own overhead, not the neural engines.

| Row | Budget | p50 | p95 |
|---|---|---|---|
| VAD endpoint hangover | 160 ms | 170 ms | 174 ms |
| Emotion fusion (parallel) | 40 ms | 3.6 ms | 4.6 ms |
| ASR final decode | 220 ms | 0.4 ms | 1.6 ms |
| Emotion state | 10 ms | 0.4 ms | 0.6 ms |
| Translation, final pass | 120 ms | 0.1 ms | 0.2 ms |
| Emotion controller | 5 ms | 0.0 ms | 0.1 ms |
| TTS first chunk | 160 ms | 5.6 ms | 11.6 ms |
| Playout buffer | 60 ms | 0.4 ms | 0.6 ms |
| **End to end** | **< 800 ms p95** | 186 ms (p95) | |

The same run met the other success targets too: ECS 0.81 (≥ 0.75), peak RAM 53 MB (< 3 GB) and
0 dropouts. The hangover row is 5 × 32 ms detector windows plus up to one window of detection
phase.

## Emotion model (p.3)

Every stage speaks V·A·D (`core/emotion/emotion_types.hpp`). The label prototypes are read off
the blueprint's map and shared with `ml/` through `config/emotion_space.json`. The controller
reproduces the walkthrough plan (pitch +15 %, range +30 %, rate +10 %, energy +4 dB, pause 80 ms,
accent ×1.4); `tests/prosody/test_controller.cpp` checks it. `emotionedge demo` runs the whole
walkthrough end to end, and `tests/pipeline/test_pipeline_e2e.cpp` checks every step of it.
