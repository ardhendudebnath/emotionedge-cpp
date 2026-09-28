# emotionedge-cpp

Real-time, emotion-preserving speech translation in C++. Speech flows through ASR, emotion
detection (valence / arousal / dominance), emotion-aware translation and expressive TTS in one
C++ process, with no Python at runtime. It is built for the edge, with a < 800 ms p95 target
from the moment the speaker stops to the first translated audio.

The design follows the *EmotionEdge-C++ Architecture* blueprint (v2). [docs/architecture.md](docs/architecture.md)
maps every box of it to code.

## Status

Roadmap phases 1–3 are done: the baseline pipe, emotion in (emotion2vec+, DistilRoBERTa, an
emotion-token NLLB LoRA) and emotion out (Kokoro-82M with learned style offsets). Phase 4 is in
progress: the models run on a GPU when there is one (see [Real engines](#real-engines)).

- **Runtime (done).** Lock-free SPSC queues, a YAML-configured stage graph, thread-per-stage
  scheduling with core pinning, and backpressure that sheds stale partials but never audio.
  Also telemetry (latency budget, ECS, Prometheus, Perfetto) and a SHA-256-verified model
  registry.
- **Emotion path, end to end (done).** Prosody features, gated late fusion, the state tracker
  with hysteresis and emphasis, the MT control tokens, the controller's rules v1, and the ECS
  closed loop.
- **Stand-in engines (built in).** Scripted ASR, a phrasebook translator, and a formant voice
  that really renders the prosody plan. With these the whole pipeline runs, and is tested,
  without model files.
- **Real engine adapters (off by default).** whisper.cpp, ONNX Runtime (Silero VAD and an
  emotion2vec head), CTranslate2 (NLLB-200), Piper, and miniaudio for live I/O. Each is
  enabled with its CMake option.
- **Offline ML factory.** `ml/` holds the P1–P5 scripts, including the P5 quality gate.

## Quick start

It needs a C++20 compiler (GCC 12+, Clang 16+ or MSVC 2022) and CMake 3.24+. GoogleTest,
nlohmann/json and yaml-cpp are fetched at pinned, hash-verified versions, or come from vcpkg.

```bash
cmake --preset release
cmake --build build/release -j
ctest --test-dir build/release --output-on-failure
```

Run the blueprint's page-3 walkthrough end to end:

```bash
build/release/apps/emotionedge demo
```

```text
utterance 1  [0.38-2.11 s]
  1 HEARD       "I can't believe you did this!"
  2 FELT        anger    V -0.24  A +0.74  D +0.71  confidence 0.58  emphasis "believe"
  3 MT INPUT    <emo=anger a=0.74 reg=casual> I can't <em>believe</em> you did this!
  4 TRANSLATED  मुझे यकीन नहीं हो रहा कि तुमने ऐसा किया!   emphasis "यकीन"
  5 PROSODY     pitch +10%  range +19%  rate +7%  energy +2.6 dB  pause 75 ms  accent x1.30
  6 ECS         0.91 (target >= 0.75, 1 clause)
```

The demo synthesizes an angry source utterance, then detects the emotion from its prosody and
words. It finds the stressed word from energy peaks and translates with emotion control tokens.
It carries the emphasis to the Hindi word, plans the prosody, synthesizes it, and re-scores the
result (ECS).

Outputs land in `out/demo/`: `translated.wav`, `captions.srt` with emotion tags, `session.json`
and `metrics.prom`. Try `--conversation` (anger, sadness, joy), `--target es` or `--realtime`.
[apps/README.md](apps/README.md) covers the CLI.

### Latency benchmark

```bash
build/release/bench/ee_bench --repeat 3
```

It paces a synthetic conversation in real time and reports every row of the blueprint's latency
budget and its success targets.

On the stand-in engines this measures the runtime's own overhead, not the neural engines:

| | Measured | Target |
|---|---|---|
| End-to-end latency, p95 | 186 ms | < 800 ms |
| Emotion consistency (ECS) | 0.81 | ≥ 0.75 |
| Peak RAM | 53 MB | < 3 GB |
| Audio dropouts | 0 | 0 / hour |

## Real engines

```bash
cmake --preset engines -DEE_ONNXRUNTIME_ROOT=/path/to/onnxruntime   # also needs CTranslate2 + espeak-ng
python models/fetch_models.py --pair en-hi                          # whisper, Silero, Piper (+ convert NLLB)
build/engines/apps/emotionedge run --input speech.wav --config config/pipeline.engines.yaml
# no microphone recording at hand? speak test input with a Piper voice:
build/engines/apps/emotionedge say --engine piper --model-id tts.piper.en_US.lessac.medium \
    --text "I can't believe you did this! | I'm so sorry, I didn't mean to hurt you." --out speech.wav
```

Build CTranslate2 with oneDNN (`-DWITH_DNNL=ON`, apt `libdnnl-dev`; see the `engines` CI job) or
MKL. Its Ruy fallback runs NLLB INT8 about 1.7x slower on x86.

Measured on CPU only (Core Ultra 9 275HX under WSL2) with Silero v6, whisper base q5_1, NLLB-200
distilled-600M INT8 (oneDNN) and Piper `hi_IN-pratham-medium`, translating English into Hindi.
Inputs were whisper.cpp's `jfk.wav` (11 s, 4 utterances) and 10 s of Piper speech (3 utterances).
All 7 transcripts matched the speech.

| Row | Budget | p50 offline / real time | p95 offline / real time |
|---|---|---|---|
| ASR final decode | 220 ms | 174–178 / 231–252 ms | 218–364 / 356–477 ms |
| Translation, final pass | 120 ms | 260–365 / 291–389 ms | 373–504 / 406–479 ms |
| TTS first chunk | 160 ms | 87–162 / 97–158 ms | 135–166 / 141–186 ms |
| End to end | 800 ms (p95) | 520–696 / 795–942 ms | 795–890 / 1065–1229 ms |

Peak RAM is 1.2 GB. The remaining gap is NLLB-600M: a CPU decodes it at 160–310 ms per
sentence, even with MKL. Faster final passes need a GPU/NPU, a smaller or distilled MT model, or
shorter segments.

Phase 2 adds emotion2vec+ (voice) and DistilRoBERTa (words) to the emotion engine. With them,
"I'm so sorry, I didn't mean to hurt you." reads as sadness (V −0.51, A −0.33, D −0.25).
The prosody rules had called it anger.
- The endpoint emotion estimate takes ~140 ms, in parallel with the ASR final decode, so it
  adds nothing to end-to-end latency.
- Peak RAM is 1.7 GB: one emotion2vec+ session is shared by the emotion engine and the
  consistency check.

Phase 2 also adds an emotion-token LoRA for NLLB, which translates `<emo=… a=… reg=…> text`.
- It scores FLORES chrF 56.7 against 55.8 for plain NLLB-600M, and uses the casual Hindi
  register the expressivity profile asks for.
- Its final pass is faster: 207–252 ms p50.
- End to end, offline p95 is 762–827 ms and real-time p50 is 696–860 ms.

Phase 3 makes Kokoro-82M the TTS. It costs latency on CPU; the vocoder is 89% of its time.
- **First chunk:** ~270–315 ms p50 at 6 threads, against 80–160 ms for Piper.
- **End to end on jfk.wav:** 1130 ms p50 (Piper: 860 ms).
- **Keeping it in check:** the first clause is capped at 3 words, and Kokoro's ~0.5 s of silence
  around each clause is trimmed.
- **Barge-in is off in this config.** It used to cancel the last translation whenever the same
  speaker kept talking. Translations now queue, and Hindi runs 1.1–1.3× the English, so fast
  speech builds a playout queue: the "Playout buffer" row.

Phase 4 runs the neural models on a GPU. The engines config sets `pipeline.device: auto`:
Kokoro, emotion2vec+, ECAPA and NLLB then run on CUDA when the build has it, and on the CPU
otherwise. DistilRoBERTa stays on the CPU, where its INT8 graph is faster (4 vs 10 ms), and
whisper.cpp runs on the CPU unless it is built with CUDA. Real-time runs, CPU forced
(`--set pipeline.device=cpu`) → GPU, on an RTX 5070 Ti laptop GPU:

| Row | Budget | p50 ms | p95 ms |
|---|---|---|---|
| Emotion fusion | 40 ms | 227–236 → 45–61 | 271–323 → 70–114 |
| ASR final decode (CPU in both) | 220 ms | 248–274 → 203–207 | 405–444 → 356–455 |
| Translation, final pass | 120 ms | 266–340 → 61–71 | 344–462 → 93–95 |
| TTS first chunk | 160 ms | 340–373 → 63–71 | 397–429 → 93–95 |
| End to end, jfk.wav | 800 ms (p95) | 1262 → **549** | 1652 → **795** |
| End to end, 3 back-to-back sentences | 800 ms (p95) | 1786 → 1196 | 3375 → 2851 |

- **jfk.wav meets the p95 target.** The run uses 20 s of CPU time instead of 51 s.
- **Back-to-back speech still misses it.** The playout queue dominates there; adaptive pacing is
  next.
- **Same output quality:**
  - NLLB with INT8/FP16 on CUDA scores FLORES chrF 56.70, against 56.72 for INT8 on the CPU.
  - emotion2vec+ on CUDA matches the CPU to 0.002 in V·A·D.
  - Kokoro's export lost a per-phoneme ONNX `Loop` that took half its GPU time. Its durations
    are unchanged.

To build for CUDA, use ONNX Runtime's GPU package and a CTranslate2 built with
`-DWITH_CUDA=ON -DWITH_CUDNN=ON`:

```bash
cmake --preset engines -DEE_ONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-gpu_cuda12-1.30.0 \
    -DCMAKE_PREFIX_PATH=/path/to/ctranslate2-cuda
# CUDA 12 + cuDNN 9 runtime libraries on the loader path, e.g. from pip:
#   nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-curand-cu12
```

The measurements linked the `libctranslate2` shipped in the `ctranslate2==4.8.2` wheel. It is
built with CUDA 12 and uses the same C++ ABI as GCC 13, so the v4.8.2 headers work with it.

| CMake option | Adds | Needs |
|---|---|---|
| `EE_WITH_WHISPER` | whisper.cpp streaming ASR | fetched automatically |
| `EE_WITH_ONNXRUNTIME` | Silero VAD, emotion2vec head, ORT execution providers | ONNX Runtime release (`EE_ONNXRUNTIME_ROOT`) or vcpkg |
| `EE_WITH_CTRANSLATE2` | NLLB-200 via CTranslate2 + SentencePiece | CTranslate2 install, SentencePiece |
| `EE_WITH_PIPER` | Piper voices | ONNX Runtime, espeak-ng |
| `EE_WITH_MINIAUDIO` | `emotionedge live` (mic → speaker) | fetched automatically |

[models/README.md](models/README.md) lists every model with its license. NLLB-200 weights are
non-commercial, and several emotion datasets are research-only: check before shipping.

## Layout

```
core/            C++20 runtime (one static library per module)
  runtime/       frames, SPSC queues, scheduler, YAML config, model registry, ORT runtime
  audio/         capture, ring, DSP, VAD & segmenter, playback, speaker encoder
  asr/           streaming ASR, LocalAgreement-2, whisper.cpp adapter
  emotion/       prosody, acoustic, lexical, fusion, state tracker (3.1), consistency (5.2)
  translate/     control tokens, glossary, wait-k, emphasis, CTranslate2 adapter
  prosody/       emotion controller, expressivity profiles, style vectors
  tts/           clause chunker, formant stand-in, Piper adapter
  telemetry/     metrics, latency budget, ECS, Perfetto trace
  pipeline/      stage registry, outputs (5.3), session runner, demo input
apps/            emotionedge CLI
bench/           latency harness (ee_bench)
config/          pipeline graphs, expressivity profiles, emotion space, phrasebook
models/          manifest.json + fetch script (model files are not committed)
ml/              offline Python factory: data, train, distill, export, eval
tests/           GoogleTest unit + end-to-end "golden audio" tests
docs/            blueprint -> code map
```

## Development

```bash
cmake --preset debug && cmake --build build/debug -j && ctest --test-dir build/debug
cmake --preset asan  && cmake --build build/asan -j  && ctest --test-dir build/asan    # ASan + UBSan
cmake --preset tsan  && cmake --build build/tsan -j  && ctest --test-dir build/tsan    # ThreadSanitizer
python -m unittest discover -s ml/tests
```

Offline runs (`emotionedge run`, `demo`, and the tests) use a deterministic single-thread
scheduler, so results are reproducible. `--realtime` and `live` run one thread per stage group,
as in the blueprint's threading model. After an intentional change to the audio output,
regenerate the golden features with `EE_UPDATE_GOLDEN=1 ctest -R PipelineE2E`.

## Roadmap

1. **Baseline pipe** (done). Capture → VAD → whisper.cpp → NLLB → Piper, with telemetry from
   day one.
2. **Emotion in** (done). emotion2vec+ and the DistilRoBERTa lexical model replace the stand-ins,
   plus an NLLB-600M LoRA that reads the emotion control prefix. All are the defaults in
   `config/pipeline.engines.yaml`; `ml/README.md` has their scores.
3. **Emotion out** (done). Kokoro-82M (StyleTTS2 family) with controller-driven durations, F0
   and contour, learned per-emotion style offsets, and the ECAPA-TDNN voice print. See
   `ml/README.md` for what transfers: arousal and dominance, not yet valence.
4. **Closed loop & speed** (in progress). GPU execution (done: CUDA for ORT and CTranslate2,
   `pipeline.device: auto`). Next: adaptive pacing of the playout queue, speaker-aware barge-in,
   whisper on the GPU, an ECS-trained controller and emotion-faithful MT data.
5. **Ship.** Desktop app, gRPC server, Android and Jetson builds, public benchmark report.

Every phase ends with the P5 quality gate (`ml/eval/quality_gate.py`).
