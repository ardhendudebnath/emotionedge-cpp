# EmotionEdge benchmark

Measured 2026-10-31 06:21 UTC at commit `ac3feb3d3d7f`, translating English into Hindi with `config/pipeline.engines.yaml`. Ranges are the lowest and highest of 5 runs; where a median is given, it comes first. Every run is in `benchmark.json` next to this file. Regenerate both with `ml/eval/benchmark.py` (its docstring has the commands).

| Machine | |
|---|---|
| CPU | Intel(R) Core(TM) Ultra 9 275HX, 24 threads, 15.3 GB RAM |
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop GPU, 595.79, 12227 MiB |
| OS | Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.39 |
| Build | emotionedge 0.1.0 |

## Blueprint targets

| Target | Goal | cpu | auto |
|---|---|---|---|
| End to end p95, jfk.wav (real time) | < 800 ms | 1065 (1040–1065) ms, **missed** | 496 (438–578) ms, met |
| End to end p95, 3 back-to-back sentences (real time) | < 800 ms | 1229 (1222–1287) ms, **missed** | 677 (657–696) ms, met |
| ASR real-time factor p95, jfk.wav | < 0.3 | 0.505–0.531, **missed** | 0.109–0.126, met |
| Peak RAM, real-time runs | < 3 GB | 2.2 GB, met | 1.9–2.0 GB, met |
| Audio dropouts, real-time runs | 0 | 0 samples, met | 0 samples, met |
| Emotion consistency (ECS), RAVDESS | ≥ 0.75 | – | 0.822–0.827, met |

## Latency, real time: jfk.wav: 11 s, 4 utterances with pauses

| Row | Budget | cpu p50 | cpu p95 | auto p50 | auto p95 |
|---|---|---|---|---|---|
| VAD endpoint hangover | 160 ms | 166 | 172–173 | 166 | 172 |
| Emotion fusion | 40 ms | 87–166 | 158–207 | 27–35 | 59–115 |
| ASR final decode | 220 ms | 260–283 | 337–348 | 4–32 | 103–248 |
| Emotion state | 10 ms | 1–5 | 4–25 | 1–5 | 5–17 |
| Translation, final pass | 120 ms | 207–211 | 274–288 | 75–93 | 99–128 |
| Emotion controller | 5 ms | 0 | 0 | 0 | 0 |
| TTS first chunk | 160 ms | 274–315 | 345–389 | 63–67 | 99–112 |
| Playout buffer | 60 ms | 0–3 | 58–154 | 0–3 | 0–7 |
| **End to end** | 800 ms (p95) | **958–1008** | **1040–1065** | **340–389** | **438–578** |

- **cpu:** CPU time 4.5–4.7 s per second of audio (model loading included), GPU memory 0 MB, ASR WER 0.000. At the start of the runs: load average 1.1–2.0, GPU at 53–57 °C
- **auto:** CPU time 0.7–0.8 s per second of audio (model loading included), GPU memory 2950–3480 MB, ASR WER 0.000. At the start of the runs: load average 0.8–1.7, GPU at 55–56 °C

## Latency, real time: 3 back-to-back sentences: 10 s, Piper's voice without its noise

| Row | Budget | cpu p50 | cpu p95 | auto p50 | auto p95 |
|---|---|---|---|---|---|
| VAD endpoint hangover | 160 ms | 178 | 178 | 178 | 178 |
| Emotion fusion | 40 ms | 154–174 | 186–206 | 4–32 | 27–51 |
| ASR final decode | 220 ms | 174–373 | 222–380 | 12–28 | 23–33 |
| Emotion state | 10 ms | 1–56 | 1–75 | 0–12 | 0–15 |
| Translation, final pass | 120 ms | 244–260 | 315–324 | 81–110 | 105–121 |
| Emotion controller | 5 ms | 0 | 0 | 0 | 0 |
| TTS first chunk | 160 ms | 324–340 | 324–404 | 64–67 | 95–109 |
| Playout buffer | 60 ms | 0–5 | 299–386 | 0–5 | 296–332 |
| **End to end** | 800 ms (p95) | **1098–1196** | **1222–1287** | **389–422** | **657–696** |

- **cpu:** CPU time 4.3–4.5 s per second of audio (model loading included), GPU memory 0 MB, ASR WER 0.000. At the start of the runs: load average 1.9–2.5, GPU at 57–58 °C
- **auto:** CPU time 0.7–0.8 s per second of audio (model loading included), GPU memory 2952–3480 MB, ASR WER 0.000. At the start of the runs: load average 0.8–0.9, GPU at 54 °C

## Quality

| Measure | Data | Result |
|---|---|---|
| ASR WER | RAVDESS, 48 clips | 0.042 |
| ECS, mean | RAVDESS, 48 clips | 0.822–0.827 |
| ECS ≥ 0.75, share of clauses | | 0.78–0.80 |
| Input emotion read correctly | | 0.65 |
| Output heard as the input's emotion | | 0.39–0.42 |
| Output heard as non-neutral | | 0.24–0.28 |
| Arousal CCC, input vs output | | 0.07–0.13 |
| Emotion from the voice (emotion2vec+): UAR / macro F1 | RAVDESS, 1440 items | 0.801 / 0.733 |
| Emotion from the voice (emotion2vec+): CCC V / A / D | | 0.86 / 0.84 / 0.83 |
| Emotion from the words (DistilRoBERTa): UAR / macro F1 | MELD test, 2610 items | 0.431 / 0.408 |
| Emotion from the words (DistilRoBERTa): CCC V / A / D | | 0.39 / 0.41 / 0.28 |
| Translation chrF / BLEU (cpu) | FLORES-200 devtest eng→hin | 56.72 / 31.47 (prefix leaks 0.000) |
| Translation chrF / BLEU (auto) | FLORES-200 devtest eng→hin | 56.70 / 31.35 (prefix leaks 0.000) |

## Runtime overhead (stand-in engines, `ee_bench`)

End to end p95 182.3–204.9 ms, peak RAM 53–54 MB, dropouts 0: the scheduler, queues and DSP without neural models.

## Models

| Model | Precision | SHA-256 | License |
|---|---|---|---|
| `vad.silero.v6` | fp32 | `1a153a22f450` | MIT |
| `speaker.ecapa.voxceleb` | fp32 | `f4f3668ae062` | Apache-2.0 (speechbrain/spkrec-ecapa-voxceleb) |
| `asr.whisper.base.q5_1` | q5_1 | `422f1ae452ad` | MIT |
| `emotion.acoustic.emotion2vec_plus_base` | fp32 | `6bc236251ec3` | FunASR model license (emotion2vec/emotion2vec_plus_base): attribution required |
| `emotion.lexical.distilroberta_en` | int8 | `327eb0811222` | none stated (j-hartmann/emotion-english-distilroberta-base): clear before shipping |
| `mt.nllb200.distilled600m.emo.int8` | int8 | `dd18b774814a` | CC-BY-NC-4.0 (non-commercial) |
| `tts.kokoro.82m.hi` | fp32 | `0e50eebba3bc` | Apache-2.0 (hexgrad/Kokoro-82M) |

## How it was measured

- **Real time** feeds the input at speaking pace through the threaded pipeline (`--realtime`), as a microphone would. End to end runs from the moment the speaker stops (the VAD endpoint) to the first translated audio.
- **Devices:** `cpu` forces every model onto the CPU; `auto` puts whisper, NLLB, Kokoro, emotion2vec+ and ECAPA on the GPU when the build has CUDA.
- **The machine's state matters.** On this laptop the same binary and config measured 1127–1294 ms p95 on jfk.wav in one session and 614–631 ms in another, taking 3–5× the CPU time for the same work. Each device gets an unrecorded warm-up run first, and every run logs the load average and the GPU's temperature as it starts. Keep the machine otherwise idle, and on mains power: with other work running (load average up to 3.9, another 2.9 GB on the GPU in one run), back-to-back sentences on the GPU measured 1004–1840 ms p95, and 926–975 ms on the quiet machine.
- **Peak RAM and CPU time** are the process's own (`wait4`). GPU memory is the device's use during the run, less its use before (`nvidia-smi`), so other processes can inflate it.
- **The emotion-transfer set** is 48 RAVDESS clips (4 actors × 8 emotions + neutral, one sentence). The emotion is in the voice only. One judge, emotion2vec+, reads both the input and the translated speech, so ECS is lenient: a neutral output still scores well. The label agreement, non-neutral share and arousal CCC show what actually transfers.
- **Translation** is scored with CTranslate2's Python API on the runtime's model files, prefix and beam; the other numbers come from the C++ runtime.
- **Small sets:** jfk.wav, three sentences and 48 clips. Between identical runs, per-emotion ECS moves by up to 0.024 and the arousal CCC by up to 0.05.
