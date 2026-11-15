# ml/: offline ML factory (Python)

Trains, compresses and exports the models the C++ runtime loads. It is never shipped to the
device (blueprint: "Python exists only in the offline factory").

| Step | Blueprint | Here |
|---|---|---|
| P1 Data | IEMOCAP, MSP-Podcast, CREMA-D, ESD; augment noise/RIR/speed | `data/prepare_manifest.py` |
| P2 Train & fine-tune | emotion2vec head → V·A·D; NLLB LoRA + emotion tokens | `train/train_vad_head.py`, `train/finetune_nllb_lora.py` |
| P3 Compress | distillation, INT8 PTQ/QAT, pruning, ONNX simplify | `distill/quantize_onnx.py`; INT8 kept only where it does not change decisions (see the exports) |
| P4 Export | torch.onnx / Optimum, CT2 converter, GGML quantize, sign + write manifest | `export/*`: `export_emotion2vec_onnx.py`, `export_lexical_onnx.py`, `export_acoustic_onnx.py`, … |
| P5 Evaluate | WER · COMET · BLEU · emotion F1 · CCC · ECS · latency | `emotionedge_ml/metrics.py`, `eval/eval_emotion.py`, `eval/quality_gate.py`, `eval/benchmark.py` |

## Contracts with the C++ core

- **Emotion space.** `config/emotion_space.json` holds the label prototypes and the ECS formula.
  Tests in both languages check that the file matches the compiled values.
- **Emotion classifiers** (`export/export_emotion2vec_onnx.py`, `export/export_lexical_onnx.py`).
  Each is a directory: `model.onnx` gives logits, and `labels.json` gives each class a V·A·D
  position (`emotionedge_ml.emotion_space.label_map`). The runtime turns the class probabilities
  into one V·A·D point with per-axis confidence (`core/emotion/class_mapping.cpp`). The acoustic
  model takes `waveform` [1, samples] at 16 kHz. The lexical model takes `input_ids` and
  `attention_mask`, plus the `tokenizer.json` that `core/emotion/bpe_tokenizer.cpp` reads.
- **Acoustic V·A·D regressor** (`export/export_acoustic_onnx.py`, a head trained by
  `train/train_vad_head.py` once dimensional data such as MSP-Podcast is available). Input
  `waveform`; outputs `vad` [1, 3] and `confidence` [1, 3].
- **MT control tokens** (`train/finetune_nllb_lora.py`). The training source carries the same text
  prefix the runtime emits, e.g. `<emo=anger a=0.8 reg=casual>`. Run the fine-tuned model with
  `--set translate.control_tokens=on --set translate.arousal_step=0.1`; this is the default in
  `config/pipeline.engines.yaml`.
- **Model registry.** `export/write_manifest.py` records size and SHA-256 in
  `models/manifest.json`, which `core/runtime/model_registry.cpp` verifies before loading.

## Quality gate (P5)

"Every phase ends with the P5 quality gate: WER, COMET, emotion F1, ECS and latency must not
regress."

```bash
build/release/bench/ee_bench --json out/bench/bench.json          # latency, ECS, RAM, dropouts
python ml/eval/quality_gate.py --bench out/bench/bench.json --candidate metrics.json
```

`eval/baseline.json` holds the current baseline. The gate fails on a regression past each metric's
tolerance, or on a miss of the blueprint's success targets (p95 < 800 ms, RTF < 0.3, ECS ≥ 0.75,
RAM < 3 GB, 0 dropouts).

Emotion models are scored through the runtime itself (`emotionedge emotion`), so the C++
inference path is what gets measured:

```bash
python ml/eval/eval_emotion.py prepare-ravdess --zip Audio_Speech_Actors_01-24.zip --dir data/ravdess --out data/ravdess.jsonl
emotionedge emotion --config config/pipeline.engines.yaml --manifest data/ravdess.jsonl --out preds.jsonl
python ml/eval/eval_emotion.py score --items data/ravdess.jsonl --preds preds.jsonl --modality acoustic --json metrics.json
```

| Model (C++ runtime) | Data | UAR | CCC V / A / D |
|---|---|---|---|
| prosody rules (phase-1 stand-in) | RAVDESS speech, 1440 clips | 0.18 | 0.00 / 0.43 / 0.03 |
| **emotion2vec+ base** | RAVDESS speech, 1440 clips | **0.80** | 0.86 / 0.84 / 0.83 |
| lexicon (phase-1 stand-in) | MELD test text, 2610 utterances | 0.21 (abstains on 53%) | 0.23 / 0.20 / 0.12 |
| **DistilRoBERTa** (INT8) | MELD test text, 2610 utterances | **0.43** | 0.39 / 0.41 / 0.28 |

How these are scored:
- UAR is over the blueprint labels: a prediction is the label nearest the model's V·A·D point.
- CCC compares against each gold label's prototype. That is a proxy, since both sets are categorical.

Caveats:
- emotion2vec+ has no "calm" class, so RAVDESS calm lands on neutral.
- RAVDESS may be in emotion2vec+'s pseudo-labelling seed data, so its UAR may be optimistic.
- DistilRoBERTa was trained on MELD's training split, not its test split.

## Benchmark report (P5)

`eval/benchmark.py` re-measures, in one command, what the gate and the README quote. It runs on
the real engines, on each device, several times over, and writes it all up with the machine and
the model hashes:

```bash
python ml/eval/benchmark.py run --emotionedge build/engines/apps/emotionedge \
    --manifest models/manifest.json --data data --out out/benchmark --runs 5 --devices cpu auto \
    --python .venv/bin/python --ee-bench build/release/bench/ee_bench
python ml/eval/benchmark.py report --results out/benchmark/results.json --out docs/benchmark.md
```

- **Latency:** real-time runs of jfk.wav and three back-to-back sentences, per device. It reports
  the budget rows, end to end, ASR WER, peak RAM, CPU time and GPU memory, and the machine's state
  as each run starts.
- **Emotion transfer** on the 48 RAVDESS clips (`eval_ecs.py`), plus ASR WER.
- **Emotion recognition** on RAVDESS and MELD (`eval_emotion.py`).
- **Translation:** FLORES chrF/BLEU (`eval_mt.py`).
- **Runtime overhead:** `ee_bench` on the stand-in engines.

`results.json` keeps every run. The report gives each value as a range, with the median for the
latency targets. [docs/benchmark.md](../docs/benchmark.md) is the current one.

## NLLB emotion-token LoRA (P2)

```bash
python ml/data/build_mt_corpus.py --meld meld_train_sent_emo.csv meld_dev_sent_emo.csv \
    --goemotions goemotions_simplified_train.parquet --out data/mt_corpus.jsonl    # 38,647 items
python ml/data/teacher_translate.py --corpus data/mt_corpus.jsonl --out data/mt_corpus.hi.jsonl
python ml/train/finetune_nllb_lora.py --data data/mt_corpus.hi.jsonl --out runs/nllb-emo-lora
ml/export/convert_nllb_ct2.sh runs/nllb-emo-lora/merged models/mt/nllb-200-distilled-600M-emo-int8 \
    mt.nllb200.distilled600m.emo.int8
python ml/eval/eval_mt.py --model models/mt/nllb-200-distilled-600M-emo-int8 --name emo --prefix runtime \
    --flores flores200_dataset --corpus data/mt_corpus.hi.jsonl --backtranslate models/mt/nllb-200-distilled-600M-int8
```

The corpus:
- **Sources:** English utterances from MELD train/dev (TV dialogue) and single-label GoEmotions.
- **Prefixes:** the exact runtime prefix, including the neutral fallback.
- **Targets:** Hindi from NLLB-1.3B, which reads plain English and so never sees the emotion.

Training is LoRA r=16 on attention and FFN (1.4% of weights): 2 epochs in 29 minutes on an RTX
5070 Ti laptop GPU, with validation loss going from 1.14 to 0.34. Results for NLLB-600M INT8, fed
the way the C++ adapter feeds it:

| | FLORES-200 devtest chrF / BLEU (human refs) | in-domain chrF (teacher refs) | prefix leaks | emotion round trip: agreement / text ECS |
|---|---|---|---|---|
| vanilla, plain text | 55.8 / 30.1 | 68.5 | 0 | 71.5% / 0.934 |
| **emotion LoRA, runtime prefix** | **56.7 / 31.4** | **74.2** | **0** | 72.7% / 0.940 |
| emotion LoRA, no prefix | 56.4 / 31.3 | – | 0 | – |

What the numbers show:
- **General quality:** it holds, and improves slightly from the 1.3B teacher.
- **Conversational fillers:** "Mmm." no longer becomes "मम्मी" ("mommy"), and "Hm-mmm" is
  now translated.
- **Prefix:** it never leaks.
- **Emotion signal:** round-trip agreement moves by 1.2 points on 600 items, which is within
  noise. The teacher never saw the emotion, so the tokens can carry little signal yet.
- **What would carry it:** emotionally faithful references, from human or LLM rewrites per
  emotion, as a phase-4 dataset. A local 8B LLM fell short (below).

`eval_mt.py` scores CTranslate2 on CUDA with INT8/FP16 when a GPU is present, as above; that is
the runtime's `gpu_compute_type`. `--device cpu --compute-type int8` scores the CPU runtime.
The two agree: FLORES chrF 56.72 / BLEU 31.47 on the CPU, against 56.70 / 31.35 on CUDA, with
no leaks on either.

`--beam` and `--stream-words` score other decoding settings: greedy decoding (`--beam 1`) gives
56.27 / 30.84 on the CPU, and the fast CPU profile's streaming final pass (`--stream-words 5`:
five words greedily, then the rest at beam 2 after them) 56.55 / 31.27.

`eval/decode_limits.py` sets the runtime's decode limits. It measures how long translations into
Hindi run from each of the 44 source languages the runtime maps, finds the repetition loops,
and shows what each candidate limit would cut (README, "Decode limits").

### LLM rewrites as emotion-faithful targets (P4, negative)

The teacher above never sees the emotion. The next step was Hindi targets from an LLM told each
item's emotion, arousal and register, run locally (no paid API):

```bash
llama-server -m Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf -ngl 99 -c 16384 -np 8 --port 8090 &
python ml/data/llm_translate.py --corpus data/mt_corpus.hi.jsonl --out pilot.jsonl --split valid --limit 300 \
    [--mode postedit --field tgt_pe]
python ml/eval/eval_mt_targets.py --corpus pilot.jsonl --fields tgt tgt_llm tgt_pe \
    --backtranslate models/mt/nllb-200-distilled-600M-int8
```

`eval_mt_targets.py` back-translates each set of targets to English with vanilla NLLB-600M. It
scores meaning (round-trip chrF against the English source) and emotion (DistilRoBERTa's label
on the round trip, against its label on the source and against the item's own label). The gate,
set beforehand: emotion agreement up, meaning held. Llama 3.1 8B Instruct (Q4_K_M) on the
RTX 5070 Ti, 297 held-out items, about 7 items/s:

| Targets | Round-trip chrF | Agreement with the source | … with the label | Text ECS |
|---|---|---|---|---|
| **NLLB-1.3B teacher** | **62.2** | **0.754** | **0.478** | **0.942** |
| LLM, first prompt (listed Hindi particles as examples) | 44.2 | 0.620 | 0.441 | 0.915 |
| LLM, strict prompt (add nothing) | 47.2 | 0.680 | 0.424 | 0.921 |
| LLM post-editing the teacher's Hindi | 52.6 | 0.660 | 0.444 | 0.920 |

The gate failed, so no corpus was generated.
- **First prompt:** "अरे" went into 75 of 300 lines, calm and neutral ones too ("Hm-mmm" →
  "अरे...").
- **Strict prompt:** it cut that to 13, but the translations still drift from the meaning.
- **Post-editing:** told to make the smallest change, it changed 276 of 300 lines, dropped
  content ("यहाँ!" → "यहीं") and mixed scripts.
- **Caveat:** the round trip goes back through NLLB, which may favour literal Hindi, and the
  classifier reads English. Emotion carried in Hindi that the round trip loses would not count.
- **What is still open:** a stronger model, or references written or judged by Hindi speakers.

## Emotion out: Kokoro style offsets (P2/P3)

Kokoro-82M reads everything in a neutral style. emotion2vec+ heard Piper, plain Kokoro and
controller-driven Kokoro alike as "neutral" on every RAVDESS clause. Extreme prosody moves
arousal (happy/surprised) but not anger or sadness. `train/learn_style_offsets.py` learns one
offset per emotion in Kokoro's 256-d style space:
- **Objective:** gradient ascent on emotion2vec+ through the differentiable TTS, on the GPU.
- **Radius:** Δ stays within the spread of the voices' own style vectors.
- **Intelligibility term:** Whisper-small's loss on the Hindi text, above the neutral render's.

`eval/eval_style_offsets.py` then judges held-out sentences with models that played no part in
training: audeering's dimensional model for V·A·D, and Whisper-small for CER (neutral renders:
0.48).

| offset | independent V·A·D shift | direction vs prototype | Whisper CER | shipped |
|---|---|---|---|---|
| anger | A +0.11, D +0.08 | ✅ | 0.30 | yes |
| joy | A +0.12, D +0.06 | ✅ | 0.31 | yes |
| sadness | A −0.25, D −0.20 | ✅ (valence ~0) | 0.51 | yes |
| fear | V +0.10, D −0.16 | ❌ fools the training judge only | 6.17 | no |
| surprise | A +0.09 | ✅ | 1.79 (hallucinations) | no |

Without the intelligibility term, CER rose to 0.59–0.87 and fear reached P = 1.00 with the
training judge. That is an adversarial solution. Valence barely moves for any offset
(|ΔV| ≤ 0.05): the offsets carry arousal and dominance.

End to end on RAVDESS (`eval/eval_ecs.py`, judged by emotion2vec+ in the consistency stage):
- **Piper and plain Kokoro:** 0–4% of clauses heard as non-neutral; arousal CCC in→out ≈ 0.01–0.03.
- **Kokoro with controls and offsets:** 21–31% non-neutral; arousal CCC 0.10–0.35.
- **Why a range:** the metric moves with benign changes. Trimming Kokoro's clause-edge silence
  alone moved it from 0.35 to 0.13, because the judge pools over silence. With ~50 utterances and
  one judge, it is noisy.
- **Why ECS alone misleads:** ECS stays ≈0.82 throughout, since a neutral output still scores
  1 − ‖src‖/2√3.

### Played as at runtime (correction, 2026-10)

Both scripts had judged renders with Kokoro's ~0.4–0.5 s of silent padding, which the runtime
trims. Both had also used the offset at full strength only, while the controller scales it by
how far the target reaches toward the emotion: about 0.7 on RAVDESS. Both scripts now trim as
the runtime does (`played()` in `train/learn_style_offsets.py`), and the learner trains at
`--strengths 0.7 1.0`.

**The shipped offsets, re-judged** (12 held-out sentences × 4 voices; mean CER, noisy):

| | strength 1.0: independent shift | emotion2vec+ P | strength 0.7: independent shift | emotion2vec+ P |
|---|---|---|---|---|
| anger | A +0.12, D +0.09 ✅ | 0.39 | A +0.09, D +0.07 ✅ | 0.04 |
| joy | A +0.12, D +0.06 ✅ | 0.50 | A +0.08, D +0.04 ✅ | 0.11 |
| sadness | A −0.30, D −0.24 ✅ | 0.31 | A −0.21, D −0.17 ✅ | 0.00 |

- **The independent judge still agrees:** the offsets move arousal and dominance the right way.
- **The runtime's judge barely notices:** at the strength the runtime usually applies,
  emotion2vec+ (the consistency judge) hardly hears them.

**Re-learned on trimmed audio, at strengths 0.7 and 1.0:** emotion2vec+ heard them far more
(held-out P at 0.7: anger 0.24, joy 0.23, sadness 0.38, surprise 0.32). The independent judge
did not:

| at strength 0.7 | independent shift | CER |
|---|---|---|
| anger | A +0.14, D +0.10 ✅, V +0.04 | 0.51 (shipped: 0.29) |
| joy | A −0.04, D −0.05 ❌ | 0.30 |
| sadness | V +0.12, A +0.02 ❌ | 0.96 |
| surprise | A −0.01 ❌ | 0.50 |

That is the adversarial case the independent check exists for: directions that please the
training judge without sounding more emotional to another model, and that cost intelligibility.
They were not adopted, and the shipped offsets stay.

The two judges disagree on trimmed audio. The next step needs a better arbiter than either:
listening tests, or a third judge that plays no part in training. A third judge was tried and
failed; on sadness, only listeners can decide (see "How far to trust the judges").

`eval/eval_tts_speed.py` sets the cap for the runtime's adaptive pacing. It renders 24 held-out
sentences with the 4 voices at each speed, and Whisper-small transcribes them:

| speed | median CER | mean CER, capped at 1 | CER > 0.5 | duration |
|---|---|---|---|---|
| 1.0 | 0.151 | 0.196 | 3% | 1.00× |
| 1.1 | 0.149 | 0.199 | 4% | 0.94× |
| 1.2 | 0.148 | 0.202 | 6% | 0.88× |
| **1.3** | 0.163 | 0.207 | 4% | 0.81× |
| 1.4 | 0.259 | 0.274 | 6% | 0.69× |
| 1.5 | 0.279 | 0.305 | 9% | 0.65× |

The cap is 1.3×. A plain mean is useless here: Whisper sometimes hallucinates on this Hindi, and
the first 12-sentence sweep's plain mean went 0.48 → 0.63 → 0.36 without any trend.

`export/export_kokoro_onnx.py` builds the frame → phoneme alignment from cumulative durations.
`torch.repeat_interleave` with per-phoneme counts exported as an ONNX `Loop` that ran once per
phoneme on the host, which was 83 of 172 ms per clause on CUDA. The export now fails if a `Loop`
or `Scan` appears. The re-exported model gives identical durations and lengths. Its spectra
differ from the old model's by no more than two runs of the old model differ from each other
(the decoder adds noise): 0.40–0.61 dB against 0.43–0.62 dB mean |log-mel|. CPU time is
unchanged.

## Learned prosody plan (P4, "ECS-trained controller")

`train/learn_controller.py` searches, per emotion, the four prosody controls that reach Kokoro:
- speaking rate;
- pitch shift;
- pitch range;
- final contour.

The values are on top of the shipped style offsets. The runtime rarely applies them in full:
like the offsets, a plan is scaled by how far the target reaches toward the emotion's
prototype, gated by confidence. That is about 0.7 on RAVDESS. So every candidate is rendered at
strengths 0.7 and 1.0.

The objective is the runtime consistency judge's (emotion2vec+) probability of the emotion, on
8 sentences × 4 voices × both strengths, minus a Whisper-loss penalty for lost
intelligibility. Rendering rounds durations, so the search is a derivative-free pattern search
started from the rules' values, inside speed ≤ 1.3. An emotion adopts the learned values only
where three checks pass on 12 held-out sentences × 4 voices, at both strengths:
- the consistency judge hears it better;
- an independent V·A·D model (audeering) also hears it closer to the emotion;
- Whisper's median CER holds within 0.02.

Held-out, at strength 0.7 / 1.0:

| emotion | consistency judge P | independent shift toward the prototype | median CER | held-out checks |
|---|---|---|---|---|
| anger | 0.10 → 0.13 / 0.50 → 0.77 | +0.14 → **+0.06** / +0.20 → **+0.08** | 0.179 → 0.163 / 0.170 → 0.176 | fail: fools the training judge |
| joy | 0.39 → 0.43 / 0.78 → 0.84 | +0.12 → +0.15 / +0.18 → +0.22 | 0.159 → 0.170 / 0.161 → 0.167 | pass |
| **sadness** | **0.08 → 0.53** / 0.51 → 0.79 | +0.21 → +0.27 / +0.26 → +0.39 | 0.179 → 0.185 / 0.200 → 0.191 | pass |
| fear | 0 → 0 | no change | – | no change found |
| surprise | 0.03 → 0.08 / 0.04 → 0.11 | +0.06 → +0.13 / +0.08 → +0.17 | 0.170 → 0.167 / 0.167 → 0.167 | pass |

The learned values at full strength:
- sadness: the rules' speed (0.94), pitch +1.4 st, range 0.5, a rising end;
- joy: speed 0.97, +4 st, range 1.5, a falling end;
- surprise: speed 1.1, +4 st, range 2.0, a rising end.

**End to end** is the last check. The 48 RAVDESS clips are run through the engines pipeline
with and without the plan, 4 runs each. The consistency stage judges every output clause.

| clauses read as sad (6 per run) | rules | learned |
|---|---|---|
| output valence | +0.06 … +0.10 | **−0.15 … −0.30** |
| output arousal | 0.16 … 0.27 | 0.20 … 0.28 |
| output dominance | −0.02 … −0.03 | **−0.23 … −0.32** |
| heard as sadness / fear / neutral (last 2 runs) | 0 / 0 / 4–5 | 1 / 2–3 / 2 |

- **Sadness** gains in V·A·D. Its ECS over all sad clips rises from 0.811–0.822 to
  0.816–0.834, and the share of output clauses heard as non-neutral from 0.20–0.24 to
  0.24–0.28. Arousal does not fall, though, so the label mostly becomes fear, not sadness: the
  plan moves valence and dominance, but not yet arousal. (The held-out numbers above were judged
  on padded renders; see the correction below.)
- **Surprise** changes nothing measurable; the judge hears neutral either way.
- **Joy** shows no gain. Its ECS was 0.779–0.794 with its plan and 0.785–0.825 without, but
  identical configurations differ by up to 0.024 between runs.
- Label agreement and the input/output arousal CCC stay within run-to-run noise. On the clauses
  the plan touches, the CCC is −0.16 in both arms.

So `config/pipeline.engines.yaml` applies the plan only to sadness (`plan_emotions`), the one
emotion that gained beyond noise. The runs above applied it to sadness and surprise, or to all
three; the plans act per emotion, and the sadness rows agree across both.

The first search only scored full strength, and its sadness plan failed end to end. On held-out
sentences it raised P(sad) from 0.51 to 0.73, and the independent judge agreed. But at 0.7 the
judge heard fear about as often as sadness. In the pipeline, all 6 sad clauses came out as fear,
with arousal +0.50.

Anger is the opposite case to sadness. The consistency judge liked a low, flat voice that the
independent judge heard as less aroused, so it keeps the rules.

### Judged as the runtime plays it (correction, 2026-10)

The numbers above came from renders the runtime never plays:
- **Padding:** Kokoro puts ~0.4–0.5 s of near-silence around every clause, and the runtime trims
  it (`core/tts/kokoro_engine.cpp`) before playback and before the consistency stage judges it.
  The script judged the padded audio.
- **Contour:** the script applied the final contour to every clause. The runtime applies it to
  an utterance's last clause only.

The padding alone made the judge hear sadness. On the RAVDESS sentence the pipeline turned to
fear, "बच्चे दरवाजे के पास बात कर रहे हैं।" (`hf_alpha`, the voice the runtime picked, strength 0.7,
8 noise seeds), the script's verdicts were:

| | Padded render | Played as at runtime | The pipeline |
|---|---|---|---|
| rules | sadness, P 1.00 | neutral | neutral, surprise |
| shipped sadness plan | sadness, P 1.00 | surprise | neutral, fear, surprise |
| a stronger plan (speed 0.8, +3.8 st) | sadness, P 1.00 | fear | fear, 11 of 12 clauses |

The script now renders as the runtime does, so it hears what the pipeline hears. It also gains
options to learn and judge on the clauses the TTS stage renders (`--clauses`), against the ECS
the runtime would report (`--objective ecs`), and to name the label the runtime would put on
each render.

**Sadness.** Re-judged held out (26 clauses × 4 voices, played as at runtime), at strength
0.7 / 1.0:

| | Heard as sad | Heard as fear | ECS | Independent shift toward sadness | median CER |
|---|---|---|---|---|---|
| rules | 0.02 / 0.19 | 0.02 / 0.03 | 0.684 / 0.701 | +0.240 / +0.322 | 0.211 / 0.244 |
| shipped plan | 0.16 / 0.65 | 0.02 / 0.06 | 0.703 / 0.879 | +0.295 / +0.429 | 0.200 / 0.250 |
| ECS search, played as at runtime | 0.26 / 0.85 | 0.07 / 0.11 | 0.731 / 0.943 | +0.342 / +0.569 | 0.250 / 0.244 |

- **The shipped plan holds**, on every measure and at both strengths, but it is far weaker than
  the padded numbers claimed: heard as sad 16% of the time at strength 0.7, not 46%. It stays.
- **The new ECS search** (speed 0.8, +2.3 st, range 0.5, rising end) hears more sadness, but
  its CER at 0.7 rises past the gate, and it adds fear. It was not adopted.
- **Strength limits the effect.** At the runtime's usual strength of 0.7, every plan barely moves
  the judge. At full strength, the shipped plan is heard as sad 65% of the time.
- **Joy and surprise**, re-judged on trimmed clauses at strength 0.7:
  - **Joy's plan:** heard as joy 26% → 36% of the time, ECS 0.802 → 0.834. It narrowly fails the
    CER gate at full strength (0.273 against 0.250 + 0.02).
  - **Surprise's plan:** heard as surprise 8% → 17%, ECS 0.774 → 0.798, the independent shift
    toward surprise +0.072 → +0.129, CER unchanged. It passes every held-out check.
  - **End to end:** the RAVDESS pipeline run with surprise added to `plan_emotions` heard the 18
    clauses read as surprise as neutral either way (ECS 0.804–0.805, 2 runs per arm). So
    `plan_emotions` stays sadness-only.
  - **The style offsets** were re-judged and re-learned on trimmed audio as well (see "Played as
    at runtime" under the style offsets above). The shipped ones stay.

## Listening test (emotion out)

The two automatic judges disagree on audio played as at runtime, so people decide.
`eval/make_listening_test.py` renders pairs of one sentence in one voice, with the same decoder
noise:
- **Plain:** no style offset, Kokoro's own prosody.
- **Emotional:** the emotion's shipped style offset and the controller's plan (the rules; for
  sadness, the learned plan), at the runtime's usual strength 0.7 and at 1.0.

Clips are rendered by the same `KokoroPlayback` the plan search judges with. They are trimmed as
at runtime and matched in loudness, so the choice rests on the voice. The sentences say nothing
emotional: RAVDESS's two statements in Hindi, and one more of the kind. The default set is 3
sentences × 2 voices (female `hf_alpha`, male `hm_omega`) × anger, joy and sadness. That is 30
pairs at 0.7, with the first two sentences also at 1.0.

Listeners pick the clip that sounds more angry, happy or sad, or "no difference". The pairs come
in a shuffled order with the sides flipped at random, per listener. The test runs as a shared
page whose answers tally across listeners. A setting carries its emotion when most people pick
its clip.

## How far to trust the judges (P4)

`eval/eval_judges.py ravdess` measures each automatic judge where the answer is known. It uses
RAVDESS's neutral, happy, sad and angry takes: 24 actors, 672 clips of acted English that none of
the judges trained on. Each clip is judged as recorded, and trimmed the way the runtime trims
Kokoro (`played()`). The judges share one interface, `emotionedge_ml/judges.py`.

The third judge is SpeechBrain's wav2vec2-base fine-tuned on IEMOCAP
(`speechbrain/emotion-recognition-wav2vec2-IEMOCAP`, Apache-2.0). It has four classes: neutral,
anger, joy and sadness. It was picked because it shares no lab, architecture or training data
with the other two judges. `judges.py` runs it with transformers alone, from the model's
checkpoint and wav2vec2-base's config. On the model's four sample clips and eight RAVDESS clips,
this matches SpeechBrain's own interface to within 1.1e-5 in every probability (SpeechBrain adds
that much noise when pooling), with the same top label on all 12.

Two measures:
- **Four-way:** balanced accuracy over the four classes. audeering is scored by the nearest
  prototype in V·A·D.
- **Pairs:** the question the TTS evaluations ask. Each actor says each sentence plainly and
  with the emotion. Is the emotional take heard as more of that emotion than the same actor's
  neutral take of the same sentence? For a classifier that means a higher P(emotion); for
  audeering, a higher projection onto the emotion's prototype. There are 192 pairs per emotion,
  and 0.5 is chance.

Trimmed clips (the recorded ones tell the same story):

| judge | four-way balanced accuracy | pairs: anger | joy | sadness (normal / strong) |
|---|---|---|---|---|
| emotion2vec+ (the training judge) | 0.92 | 1.00 | 0.99 | 0.97 (0.94 / 1.00) |
| audeering V·A·D (independent) | 0.48 | 0.99 | 0.95 | **0.60 (0.80 / 0.41)** |
| IEMOCAP (the third judge) | 0.35 | 0.78 | 0.57 | 0.63 (0.61 / 0.65) |

- **The IEMOCAP judge fails** (negative). It hears anger in 81–88% of all clips, with P 0.99 on
  average, and sadness in at most 2 of the 192 sad ones. Its pairs are barely above chance for
  joy and sadness. It does not carry over from IEMOCAP's recordings, so it cannot arbitrate, and it was
  not run on Kokoro.
- **audeering is reliable for anger and joy, not for sadness.** It hears mainly arousal and
  dominance:
  - Its valence is negative for every class. Joy comes out at −0.19 normal and −0.28 strong,
    no better than neutral's −0.20, so its four-way joy recall is 0.03.
  - RAVDESS's strong sadness is crying. It raises audeering's arousal and dominance (A +0.13,
    D +0.16, against A −0.25, D −0.07 for normal sadness), so audeering hears it as less sad
    than the neutral take.
  - Its sadness shift therefore measures quiet sadness only.
- **emotion2vec+ is by far the best on real speech.** Its RAVDESS scores may be flattered,
  since we cannot rule out that RAVDESS is in its training data. On Kokoro, though, it is the
  judge the offsets and plans were optimised against. Where it disagrees, it is the party under
  suspicion: the fear offset fooled it alone (see the style offsets above).

What this changes:
- **Anger and joy:** audeering's verdicts stand. The re-learned joy offset, which emotion2vec+
  liked and audeering heard as less aroused, stays rejected.
- **Sadness has no automatic arbiter.** audeering's support for the shipped sadness plan (an
  independent shift of +0.240 → +0.295 at strength 0.7) is weak evidence, and emotion2vec+'s
  cannot be checked. Sadness is what the listening test has to settle, through its 10 sadness
  pairs.

```
python ml/eval/eval_judges.py ravdess --ravdess ~/ee-data/ravdess --json out/judges.json
```

## Speaker-aware barge-in (P4)

`eval/eval_speaker_verification.py` calibrates the speaker stage's `VoiceGate`. The 24 RAVDESS
actors are recorded in one studio on one microphone, so voices are not told apart by their
channel. Each actor gets a reference print, built from the first 2 s of each enrolled clip:
- cold: 3 neutral/calm clips;
- warm: 8 clips of any emotion.

Probes are the first T s of every other clip, scored by ECAPA cosine. The table gives two
error rates:
- false barge-in: the actor's own speech judged another voice;
- missed: another actor judged the same voice; in brackets, same-gender pairs only.

| probe | reference | EER | at 0.20: false / missed | at 0.25: false / missed |
|---|---|---|---|---|
| 1.0 s | cold | 11.1% | 10.9% / 11.3% (20%) | 17.6% / 5.4% (11%) |
| 1.0 s | warm | 8.0% | 5.0% / 19.7% (35%) | 6.9% / 10.3% (20%) |
| **1.5 s** | **warm** | 4.2% | 0.9% / 22.7% (41%) | **1.8% / 12.6% (25%)** |
| 1.5 s | cold | 7.6% | 4.7% / 13.7% (25%) | 8.3% / 6.9% (14%) |
| 2.0 s | warm | 3.3% | 0.5% / 23.8% (43%) | 0.9% / 13.5% (27%) |

The runtime uses a 1.5 s probe, threshold 0.25, and arms once 3 utterances are enrolled.

- **Errors are not symmetric.** Cutting off the interpreted speaker's own translation is the
  costly one. Missing a listener's interruption leaves things as they were before barge-in.
- **Echo of the translation:** Kokoro's voices score ≥ 0.59 (median 0.75) against their own
  print at 1.5 s, and the actors score ≤ 0.32 against them. Echo threshold 0.45.

## Background noise (P5)

How much does a noisy room cost? `eval/eval_noise.py` mixes noise into a RAVDESS input and runs
the whole pipeline on each mix (`pipeline.engines.yaml`, on the GPU).
- **Input:** 192 clips: every actor saying "Kids are talking by the door" in 8 emotions (strong;
  neutral at normal), 1.6 s of silence between clips, 17 min in all.
- **Noise:**
  - white noise;
  - pink (1/f) noise;
  - babble: eight RAVDESS talkers at once, saying the other sentence.
  It is mixed in at 20, 10, 5 and 0 dB SNR. The SNR counts the speech power over the clips, and
  the noise runs through the gaps too.
- **Scores, per clip:**
  - missed: no utterance at all;
  - WER;
  - the emotion read from the voice: the fused label, with a missed clip counting as wrong.

```bash
python ml/eval/eval_ecs.py build --items data/ravdess.jsonl --out data/ravdess_all.wav --actors $(seq 1 24)
python ml/eval/eval_noise.py run --wav data/ravdess_all.wav --items data/ravdess.jsonl \
    --emotionedge build/engines/apps/emotionedge --config config/pipeline.engines.yaml --out out/noise
```

As shipped before this measurement (automatic language ID on every utterance):

| Input | Missed clips | WER | Emotion read right |
|---|---|---|---|
| clean | 0 | 0.076 | 0.625 |
| white 20 dB | 7 | 0.293 | 0.531 |
| white 10 dB | 18 | 0.537 | 0.359 |
| white 5 dB | 33 | 0.660 | 0.260 |
| white 0 dB | 73 | 0.807 | 0.167 |
| pink 20 dB | 7 | 0.184 | 0.578 |
| pink 10 dB | 19 | 0.392 | 0.464 |
| pink 5 dB | 37 | 0.536 | 0.359 |
| pink 0 dB | 88 | 0.705 | 0.193 |
| babble 20 dB | 70 | 3.20 | 0.318 |
| babble 10 dB | 116 | 5.01 | 0.146 |

What the runs show:
- **Even 20 dB costs a lot.** WER rises from 0.08 to 0.18–0.29, the emotion is read right less
  often (0.63 → 0.53–0.58), and 7 clips vanish.
- **The vanished clips are the VAD's.** On the raw white 20 dB input, Silero alone, fed as the
  runtime feeds it, misses 11 clips; its probability over the noise averages 0.28. Resetting its
  recurrent state just before each clip recovers most of them (0 → 38 windows above 0.5 on one),
  so the state drifts in long noise. That reset knows where the clips are, which the runtime
  doesn't. The practical fix is below ("Silero in steady noise").
- **Babble breaks endpointing.** Other people's speech never lets the VAD hear silence, so
  utterances run to the 15 s cut: a median of 14.9 s at 10 dB, against 1.8 s clean. Whisper
  transcribes the babble, which gives the WER above 1, and on those long garbled transcripts
  NLLB's decodes looped. Babble at 5 and 0 dB was not finished: one run took over 12 minutes,
  and 10 dB already shows the failure. Endpointing on the interpreted speaker's own voice (the
  ECAPA print the barge-in gate already has) would be the fix.
- **A looping decode crashed the translation.** The decode limit grows with the source (3
  tokens per source token, plus 32 for a final), so a 15 s babble transcript scaled it past
  NLLB's 1024 positions. CTranslate2 then failed at position 1024 instead of reporting a cut.
  The limit is now held to the model's positions (`max_positions`, 1024). A decode of exactly
  1024 tokens runs, and 1025 fails.
- **Language ID flips, even on clean speech.** Whisper detects each utterance's language from its
  first second, and 15 of 212 clean utterances came out Korean, German, Japanese, Hindi or
  Turkish ("Kids!" heard as Japanese). In noise there are more (40 of 200 at white 10 dB), and
  each one is transcribed in the wrong language. Pinning the language to English
  (`asr.language: en`) cuts WER from 0.076 to 0.042 clean, and by 0.03–0.05 at every noise
  level; the emotion is unaffected. Pinning gives up every other source language, though; the
  language prior below doesn't.

### The session's language prior

The asr stage keeps a language for the session (`core/asr/language_prior.hpp`). It starts in the
pipeline's `source_language`. A detection of another language moves the session there only if
whisper is at least `language_switch_p` sure of it; otherwise the utterance is transcribed in
the session's language. The engine now detects the language itself
(`whisper_lang_auto_detect_with_state`), because whisper_full keeps only the winner. The cost is
the same encoder pass whisper_full spent on it. `ee_asr_language_overruled_total` counts the
overruled detections, and `--log debug` shows each one.

**Calibration.** Two sets of whisper's detections, logged with `--log debug`:
- the RAVDESS mixes, which are all English;
- synthesised Hindi: 72 sentences from the MT corpus's held-out Hindi (24 short, 24 medium, 24
  long), spoken by Kokoro's hf_alpha and hm_omega and by Piper's pratham. A Hindi speaker in a
  session that starts in English should move it.

The rule was replayed over these at every threshold. The table gives the share of utterances
transcribed in their true language:

| Rule | English clean | English, noise 20/10 dB | Hindi (3 voices) |
|---|---|---|---|
| no prior (each utterance on its own) | 0.93 | 0.80–0.89 | 0.51–0.54 |
| **switch at p ≥ 0.7** | **0.97** | **0.91–0.95** | **0.85–0.95** |
| switch at p ≥ 0.8 | 0.98 | 0.90–0.97 | 0.56–0.88 |
| switch after two sure utterances in a row (p ≥ 0.85) | 0.99 | 0.97–0.99 | 0.58–0.70 |
| pinned to English | 1.00 | 1.00 | 0.00 |

- **Whisper base is weak at naming Hindi.** On its own it calls half the synthesised Hindi
  Arabic, Urdu, Japanese, Chinese or English. A session that has switched to Hindi holds against
  those, which is most of the Hindi gain.
- **Whisper is sometimes sure and wrong**, at up to p 0.99 on noisy English. Requiring two
  utterances in a row therefore protects English best, but whisper seldom names Hindi twice
  running, so a Hindi speaker's session seldom switched.
- **0.7 balances the two.** Thresholds from 0.65 to 0.75 score about the same. At 0.8, two of
  the three Hindi voices seldom switch.

End to end with the prior at 0.7 (the shipped default):

| Input | WER, automatic | WER, pinned to English | WER, prior |
|---|---|---|---|
| clean | 0.076 | 0.042 | 0.053 |
| white 20 dB | 0.293 | 0.247 | 0.269 |
| white 10 dB | 0.537 | 0.507 | 0.539 |
| white 5 dB | 0.660 | 0.625 | 0.661 |
| pink 20 dB | 0.184 | 0.148 | 0.168 |
| pink 10 dB | 0.392 | 0.366 | 0.392 |
| pink 5 dB | 0.536 | 0.510 | 0.539 |

The synthesised Hindi is now transcribed in Hindi for 77 of 90, 100 of 105 and 69 of 75
utterances; without the prior it was about half. The emotion and the missed clips are as
before.
- **At 10 dB and below, WER doesn't move.** The language now comes out right more often (0.80
  → 0.91 at white 10 dB). But whisper gets those words wrong in English too.
- **What's left in steady noise is the VAD's and the recognizer's.** That is the next work: the
  VAD's drifting state, then noise suppression or a larger whisper model.

### Silero in steady noise

The candidates were compared offline first, in a model of the runtime: the front end's AGC (20 ms
blocks), Silero v6 on 32 ms windows, and the segmenter's EndpointTracker. It reproduces the
pipeline's missed clips (7/18/7/19 in white and pink noise at 20/10 dB, against 7/18/7/19 end to
end), so each candidate could be tried in minutes:

| Candidate | clean | white 20 | white 10 | pink 20 | pink 10 | Utterances ending later |
|---|---|---|---|---|---|---|
| Silero as it runs (missed clips) | 0 | 7 | 18 | 7 | 19 | – |
| lower thresholds (on/off 0.3)† | 0 | 7 | 21 | 6 | 25 | – |
| reset the state after 1 s without speech† | 4 | 10 | 23 | – | – | – |
| two states reset in turn, every 8 s | 3 | 3 | 10 | 4 | 9 | – |
| **carried state + one reset every 4 s, max of the two** | 0 | 4 | 11 | 2 | 9 | 62 of 192 clean, +29 ms; +47 to +157 ms in noise |
| **… the fresh state only to start an utterance** | **0** | **4** | **10** | **2** | **9** | **1 of 192** |

† An earlier run of the model, without the AGC, which counted a clip as missed only if no
utterance touched it. In that run, Silero as it runs missed 0, 11, 27, 9 and 28.

What this shows:
- **A freshly reset state is more conservative.** On its own it loses quiet clips on clean
  speech, which a state that has heard speech catches. So the carried state stays, and the fresh
  one is added beside it.
- **It must never be reset on speech.** It is reset only between utterances, not during one or
  while an onset is being counted, and it is ignored for 500 ms after each reset while it settles.
- **It must not decide where an utterance ends.** It stays above the off threshold longer, so
  ending on the max delayed endpoints: by 29 ms on average on clean speech, by up to 157 ms in
  noise. That is latency. Used only to start an utterance, it leaves the endpoints where they were.
- **Thresholds and noise-floor detectors were weaker.** Lower thresholds recover only a few clips,
  because the missed clips' probability sits near 0, not just under 0.5. The energy detector
  helps in white noise, but at its default it fires on pink noise between the clips. Real rooms
  add door slams and keyboard clicks, which Silero is built to reject.
- **Babble is not this problem.** The AGC makes it worse: without it, babble at 20 dB lost 18–20
  clips instead of 71. But capping the gain at 6 dB still lost 60. Silero's verdict on background
  talkers flips within a few dB of level, so babble wants endpointing on the speaker's own
  voice, not AGC tuning.

The runtime's Silero keeps the two states (`refresh_s: 4`, `refresh_warmup_ms: 500`;
`core/audio/silero_detector.cpp`). The segmenter tells it its phase (idle, onset, speech) before
each window. End to end, against the build before it (both with the language prior):

| Input | Missed clips | WER | Emotion read right |
|---|---|---|---|
| clean | 0 → 0 | 0.053 → 0.056 | 0.615 → 0.615 |
| white 20 dB | 7 → **4** | 0.269 → 0.260 | 0.536 → 0.536 |
| white 10 dB | 18 → **11** | 0.539 → 0.535 | 0.354 → 0.359 |
| white 5 dB | 33 → **27** | 0.661 → 0.669 | 0.255 → 0.260 |
| pink 20 dB | 7 → **2** | 0.168 → 0.163 | 0.573 → 0.568 |
| pink 10 dB | 19 → **9** | 0.392 → 0.385 | 0.474 → 0.490 |
| pink 5 dB | 37 → **22** | 0.539 → 0.544 | 0.354 → 0.370 |
| babble 20 dB (utterances outside the clips: 25 → **170**) | 70 → 50 | 3.20 → 2.09 | 0.318 → 0.365 |

- **The missed clips fall from 121 to 75 over the six noisy inputs**, by 18% (white 5 dB) to 71%
  (pink 20 dB). Each recovered clip is an utterance that now gets translated at all.
- **Babble has a cost.** The fresh state hears the background talkers too, so 170 utterances now
  start between the clips, against 25. Each would be translated, about one every 6 s of the
  17 min. A stricter bar for the fresh state is a straight trade, measured in the model:

  | Fresh state may start an utterance at | Babble 20 dB: utterances between clips | Steady noise (4 inputs): missed |
  |---|---|---|
  | (no fresh state) | 26 | 51 |
  | p ≥ 0.5 (shipped) | 172 | 26 |
  | p ≥ 0.85 | 106 | 34 |
  | p ≥ 0.95 | 64 | 42 |

  Babble is speech, so a detector that hears speech better hears it too. Only the speaker's voice
  can tell them apart: that is the next step, and it would also drop these utterances. Until then,
  `refresh_s: 0` keeps the old behaviour in babble-heavy places.
- **WER barely moves.** Whisper base gets most of the recovered clips wrong too, so what remains
  is the recognizer's: noise suppression or a larger model.
- **On clean speech** nothing is lost and the endpoints hold: every utterance ends where it did,
  and 31 start earlier. The WER difference is 3–4 words of 1,152.

## Setup

The metrics, the gate and the manifest tools use only the standard library:

```bash
python -m unittest discover -s ml/tests
```

Training, export and full evaluation need `pip install -r ml/requirements.txt`. Check dataset
and model licenses before shipping. Several emotion corpora are research-only, and NLLB-200 is
CC-BY-NC-4.0.
