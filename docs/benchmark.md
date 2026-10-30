# EmotionEdge benchmark

Measured 2026-10-30 19:16 UTC at commit `8621e676b51d`, translating English into Hindi with `config/pipeline.engines.yaml`. Ranges are the lowest and highest of 5 runs; where a median is given, it comes first. Every run is in `benchmark.json` next to this file. Regenerate both with `ml/eval/benchmark.py` (its docstring has the commands).

| Machine | |
|---|---|
| CPU | Intel(R) Core(TM) Ultra 9 275HX, 24 threads, 15.3 GB RAM |
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop GPU, 595.79, 12227 MiB |
| OS | Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.39 |
| Build | emotionedge 0.1.0 |

## Blueprint targets

| Target | Goal | cpu | auto |
|---|---|---|---|
| End to end p95, jfk.wav (real time) | < 800 ms | 1622 (1608–1720) ms, **missed** | 607 (590–641) ms, met |
| End to end p95, 3 back-to-back sentences (real time) | < 800 ms | 1688 (1649–1949) ms, **missed** | 940 (926–975) ms, **missed** |
| ASR real-time factor p95, jfk.wav | < 0.3 | 0.493–0.698, **missed** | 0.076–0.092, met |
| Peak RAM, real-time runs | < 3 GB | 2.2 GB, met | 1.9–2.0 GB, met |
| Audio dropouts, real-time runs | 0 | 0 samples, met | 0 samples, met |
| Emotion consistency (ECS), RAVDESS | ≥ 0.75 | – | 0.822–0.827, met |

## Latency, real time: jfk.wav: 11 s, 4 utterances with pauses

| Row | Budget | cpu p50 | cpu p95 | auto p50 | auto p95 |
|---|---|---|---|---|---|
| VAD endpoint hangover | 160 ms | 166 | 172 | 166 | 172 |
| Emotion fusion | 40 ms | 95–114 | 154–170 | 20–28 | 54–79 |
| ASR final decode | 220 ms | 274–307 | 373–493 | 0–1 | 68–154 |
| Emotion state | 10 ms | 0–1 | 4–6 | 0–1 | 4–5 |
| Translation, final pass | 120 ms | 252–283 | 369–410 | 63–69 | 83–125 |
| Emotion controller | 5 ms | 0 | 0 | 0 | 0 |
| TTS first chunk | 160 ms | 307–373 | 389–471 | 55–65 | 93–128 |
| Playout buffer | 60 ms | 0 | 500–778 | 0–1 | 296–346 |
| **End to end** | 800 ms (p95) | **1130–1229** | **1608–1720** | **340–389** | **590–641** |

- **cpu:** CPU time 4.5–4.8 s per second of audio (model loading included), GPU memory 0 MB, ASR WER 0.000. At the start of the runs: load average 1.6–3.1, GPU at 57–58 °C
- **auto:** CPU time 0.6–0.7 s per second of audio (model loading included), GPU memory 2953–3479 MB, ASR WER 0.000. At the start of the runs: load average 1.4–2.7, GPU at 57–59 °C

## Latency, real time: 3 back-to-back sentences: 10 s, Piper's voice

| Row | Budget | cpu p50 | cpu p95 | auto p50 | auto p95 |
|---|---|---|---|---|---|
| VAD endpoint hangover | 160 ms | 182 | 190 | 182 | 190 |
| Emotion fusion | 40 ms | 75–112 | 97–211 | 9–13 | 20–28 |
| ASR final decode | 220 ms | 162–315 | 180–365 | 0–8 | 14–20 |
| Emotion state | 10 ms | 1–21 | 5–52 | 4–5 | 7–14 |
| Translation, final pass | 120 ms | 291–332 | 364–529 | 69–85 | 99–111 |
| Emotion controller | 5 ms | 0 | 0 | 0 | 0 |
| TTS first chunk | 160 ms | 332–381 | 379–410 | 60–85 | 91–113 |
| Playout buffer | 60 ms | 0 | 598–893 | 63–93 | 559–628 |
| **End to end** | 800 ms (p95) | **1130–1458** | **1649–1949** | **422–471** | **926–975** |

- **cpu:** CPU time 4.3–4.7 s per second of audio (model loading included), GPU memory 0 MB, ASR WER 0.000. At the start of the runs: load average 2.4–3.1, GPU at 57–59 °C
- **auto:** CPU time 0.6 s per second of audio (model loading included), GPU memory 2951–3479 MB, ASR WER 0.000. At the start of the runs: load average 0.8–1.3, GPU at 59 °C

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
