# Multi-session batching: what was measured and changed

The streaming server (`emotionedge serve`) runs one live pipeline per client. The sessions share one copy of each model, and on a GPU they batch their translations. This is the work on four questions about that, B1–B4:
- **B1:** do decode caps cut translations short?
- **B2:** do finals wait behind other work?
- **B3:** do the models get in each other's way on the GPU?
- **B4:** how many sessions fit in GPU memory?

The measurements are below. Each change shipped with its evidence; changes that didn't help were measured, written down, and left out.

All latency figures come from the real engines on one laptop (RTX 5070 Ti laptop GPU, 12 GB, on mains power, power plan "Performance"), with jfk.wav streamed at speaking pace by `bench/load_server.py`. A/Bs interleave their arms in blocks, with the same binary in both arms.

| | Question | Outcome | PR |
|---|---|---|---|
| B1 | Do decode caps cut translations short? | Yes, silently: a flat 256-token cap on finals, and loops in 19 of 44 source languages. Limits now scale with the source, every cut is reported, and cut or looping finals are retried. | #28, #34 |
| — | Where does time go with N sessions? | Load harness and baseline: throughput peaks at 10 sessions. | #29, #31 |
| B4 | How many sessions fit in GPU memory? | About 10 on 12 GB. Two automatic limits were built and measured, and both were wrong. `--max-sessions` is set from a load test. | #32 |
| B3 | Do the models get in each other's way? | Yes; emotion2vec+ is the largest GPU user under load. It is now batched across sessions. Thinning its partial estimates was ruled out. | #31, #33 |
| B2 | Do finals wait behind drafts? | Yes, when sessions start at different times. Two ways of reordering NLLB's work were measured: one helped the median but hurt the tail, the other was worse. Neither ships; arrival order stays. | this report |

## Baseline: where the time goes (#29, #31)

`bench/load_server.py` streams a WAV from N clients at once or staggered (one every 700 ms). It reports, per run:
- first audio and final caption latency, caption freshness and refusals;
- NLLB wait and run times (from a `--log debug` server log);
- GPU memory;
- each stage's busy time summed over sessions (`stage_time` in the session report): a per-model profile.

| Sessions | GPU memory, loaded / peak | First audio p50, sync / staggered | Utterances/s |
|---|---|---|---|
| 1 | 3.0–3.5 / 3.5 GB | 330 / 318 ms | 0.30 |
| 3 | 4.7 / 4.9 GB | 586 / 420 ms | 0.87 |
| 8 | 7.5–8.0 / 8.8 GB | 1,170 / 1,692 ms | 2.2–2.3 |
| 10 | 9.3 / 11.3 GB | 1,442–1,642 / 2,642 ms | **2.75** |
| 12 | 10.2 / 11.9 GB | 3,158–5,062 / 3,900 ms | 2.4–2.7 |
| 16 | 11.9 / 11.9 GB | 36 s / 9.2 s | 1.3 |

Busy time per stage at 8 synchronized sessions, against 1 session:

| Stage | 1 session | 8 sessions |
|---|---|---|
| emotion (emotion2vec+) | 0.8 s | **36 s** |
| translate (NLLB) | 0.6 s | 23 s |
| asr | 1.1 s | 20 s |
| consistency (emotion2vec+ on the output) | 0.25 s | 11 s |
| tts | 0.28 s | 9 s |

## B1: decode limits (#28, #34)

- **Root cause:** finals had a flat 256-token cap and drafts 3 × source + 32. Neither said whether the model ended a decode or the cap cut it. A batch decodes to its longest member's cap, so a result depended on what it was batched with.
- **Measured:** all 1,012 FLORES-200 devtest sentences from each of the 44 source languages the runtime maps into Hindi, as finals and as half-sentence drafts: 89,056 decodes (`ml/eval/decode_limits.py`).
  - 22 finals and 12 drafts were repetition loops.
  - Everything else stayed within 2.0 × the source (finals) and 2.62 × (drafts).
- **Changes:**
  - **Limits:** finals 3 × source pieces + 32, drafts 2.5 × + 24, held per request inside batches.
  - **Retry:** a final cut at its limit is decoded once more with repetition penalty 1.2. If that is cut too, it is flagged `truncated` in the live event, gRPC and the session export.
  - **Drafts:** a cut draft is dropped, so its words are never committed as the final's prefix.
  - **Reporting:** every cut is counted (`ee_translate_truncated_total{pass, language, outcome}`) and logged.
- **Result:** the limits cut 17 finals and 10 drafts, every one a loop, with nothing else cut (42 and 21 tokens to spare). All 17 retries ended as whole sentences.
- **Multilingual regression:** chrF into Hindi rose in all 15 languages that had a cut final; for example English 56.70 → 56.74 and Korean 42.22 → 42.40. Only the cut sentences changed. BLEU was up or equal everywhere except Khmer (15.62 → 15.60). The full table is in the README, "Decode limits".
- **Loops that end inside the limit (#34):** 5 finals and 2 drafts. A repetition check catches them: under half the output's 4-grams distinct, over at least 12 pieces. On the decodes long enough to judge it flags those 7 and none of the other 44,391 finals and 34,687 drafts. They then get what a cut decode gets, and the retry ended all 5 finals as sentences.

## B4: GPU memory and overload (#32)

- **Root cause:** past the GPU's memory the driver spills into system memory. Nothing fails and nobody is refused; every model slows at once. At 14 sessions, ASR final decode reached 10 s and emotion fusion 17 s.
- **Two automatic limits were built (unit, ASan and TSan tested) and measured. Neither holds:**
  - **Refusing new sessions on free GPU memory (NVML):** memory the engines cached in a busy round reads as used, so 5 of 8 clients were refused in a round that fit. Simultaneous connections all passed before any had allocated.
  - **Counting sessions measured idle at start:** it measured ~400 MiB per session and allowed 15–17. A running session needs ~0.45 GB more, because the shared engines' allocators grow with how many sessions run at once. 12 sessions then reached 5–6 s first audio.
- **Result:** `--max-sessions` stays the control. Set it from a load test on the server's GPU: 8–10 here. `apps/README.md` explains how.

## B3: models contending for the GPU (#31, #33)

- **Ruled out: thinning emotion2vec+'s partial estimates** (every 500 ms, on the last 1 s). The final emotion is smoothed from them. On the 48 RAVDESS clips, through the whole pipeline, the input emotion was read correctly at these rates:
  - every 500 ms (current): 0.647
  - every 1,000 ms: 0.569
  - none: 0.529
- **Batching emotion2vec+ across sessions:** the partials are all 1 s long, so they batch with no padding.
  - **Export:** a dynamic batch axis. Batched rows match single runs to 6e-6 in the logits, and alone the new export's logits are bit-identical to the old one's.
  - **Runtime:** one batcher per shared model, grouped by length. It is on only on a GPU with several sessions.
- **A/B, 8 sessions, 3 interleaved blocks:**

  | | Unbatched | Batched |
  |---|---|---|
  | Synchronized: first audio p50 / p95 | 1,502 / 1,650 ms | **1,296 / 1,453 ms** |
  | Staggered: first audio p50 / p95 | 3,022 / 4,062 ms | **2,512 / 3,278 ms** |
  | Staggered: emotion fusion p50 | **92 ms** | 228 ms |

  With staggered starts, emotion arrives later, because one worker serves the model's calls. It is off the critical path there. Run to run the spread overlaps: a likely gain, not a proven one.
- **To use it:** re-export emotion2vec+ with `ml/export/export_emotion2vec_onnx.py` (`models/README.md`). An older export runs unbatched, as before.

## B2: scheduling NLLB finals

- **Root cause:** with staggered starts, 69% of finals shared an NLLB batch with other sessions' drafts. They waited 696 ms p50 and ran 704 ms, against 321 ms for a final alone. Each slow mixed batch then held up the requests behind it. With synchronized starts, drafts and finals never mix (finals batch with finals, 142 ms).
- **Two policies were measured against arrival order (`fifo`)**, 8 staggered sessions, interleaved:

  | | fifo | finals first (finals and drafts apart, waiting finals first) | apart (finals and drafts apart, arrival order) |
  |---|---|---|---|
  | Runs | 6 / 4 | 6 | 4 |
  | First audio p50 | 1,940 / 2,041 ms | **1,775 ms** | 2,214 ms |
  | First audio p95 | **2,618 / 2,705 ms** | 3,058 ms | 3,378 ms |
  | Final caption p50 | 1,436 / 1,703 ms | **1,220 ms** | 1,826 ms |
  | NLLB final wait / run | 312 / 581 ms; 472 / 564 ms | 146 / 242 ms | 686 / 356 ms |
  | Longest caption wait p95 | **2,215 / 2,465 ms** | 3,145 ms | 2,920 ms |

  (fifo has two values: its runs in the finals-first A/B, then in the apart A/B.)

- **Finals first:** finals get much faster at NLLB. But drafts then wait (225 → 441 ms) and commit fewer words in time, so some finals decode more: the tail of first audio and of caption updates gets worse. That is the same lesson as both earlier draft-skipping experiments (`apps/README.md`).
- **Apart:** finals run faster but wait longer, behind separate draft batches instead of joining the one about to run. It is worse on both median and tail.
- **Result:** neither ships, and arrival order stays. With staggered starts, the delay comes from all the models sharing one GPU, not from the order NLLB runs its requests in.
- **Caveats:**
  - One A/B (apart) was cut short when the laptop came off mains during its third block, so it has 4 runs per arm.
  - An earlier attempt was discarded: a k3s cluster in WSL started part-way through and took the CPU and GPU (10–16 s first audio in both arms). The A/B script now checks for a quiet machine before every arm.

## Not covered, and open

- **Battery:** only mains-power runs are reported. On battery the GPU throttles 2–3× and gets worse as the battery drains, which made earlier battery runs unrepeatable. Comparisons were therefore made on mains only, and runs wait for it.
- **GPU contention with staggered starts:** NLLB slows while other sessions' ASR, emotion2vec+ and TTS share the GPU. The final translation pass is the largest step users wait on at 8 sessions. Reordering NLLB doesn't help (B2), so the lever left is the capacity limit (B4).
- **Emotion-faithful MT data:** a local 8B LLM fell short of the NLLB teacher on round-trip meaning and emotion (#30); a stronger model or Hindi speakers are needed.

## Files

- **#28:** `core/translate/decode_limits.hpp`, `ct2_translator.cpp`, `translate_stage.{hpp,cpp}`, `translator.hpp`, `phrasebook.{hpp,cpp}`, `core/pipeline/recorder.{hpp,cpp}`, `core/runtime/frame.hpp`, `core/server/grpc_server.cpp`, `proto/emotionedge/v1/translator.proto`, `ml/eval/decode_limits.py`, `tests/translate/test_translate.cpp`, docs.
- **#34:** `core/translate/decode_limits.hpp`, `ct2_translator.cpp`, `translate_stage.cpp`, `translator.hpp`, `tests/translate/test_translate.cpp`, docs.
- **#29, #31:** `bench/load_server.py`, `core/pipeline/recorder.cpp` (`stage_time`), `tests/pipeline/test_pipeline_e2e.cpp`, docs.
- **#30:** `ml/data/llm_translate.py`, `ml/eval/eval_mt_targets.py`, docs.
- **#32:** `apps/README.md`.
- **#33:** `core/emotion/onnx_acoustic.cpp`, `emotion_stage.cpp`, `consistency_stage.cpp`, `ml/export/export_emotion2vec_onnx.py`, `tests/emotion/test_classifier_models.cpp`, docs.
- **This report:** `docs/batching.md`, `apps/README.md`, `README.md`.
