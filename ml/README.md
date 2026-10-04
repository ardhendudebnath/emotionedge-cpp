# ml/: offline ML factory (Python)

Trains, compresses and exports the models the C++ runtime loads. It is never shipped to the
device (blueprint: "Python exists only in the offline factory").

| Step | Blueprint | Here |
|---|---|---|
| P1 Data | IEMOCAP, MSP-Podcast, CREMA-D, ESD; augment noise/RIR/speed | `data/prepare_manifest.py` |
| P2 Train & fine-tune | emotion2vec head → V·A·D; NLLB LoRA + emotion tokens | `train/train_vad_head.py`, `train/finetune_nllb_lora.py` |
| P3 Compress | distillation, INT8 PTQ/QAT, pruning, ONNX simplify | `distill/quantize_onnx.py`; INT8 kept only where it does not change decisions (see the exports) |
| P4 Export | torch.onnx / Optimum, CT2 converter, GGML quantize, sign + write manifest | `export/*`: `export_emotion2vec_onnx.py`, `export_lexical_onnx.py`, `export_acoustic_onnx.py`, … |
| P5 Evaluate | WER · COMET · BLEU · emotion F1 · CCC · ECS · latency | `emotionedge_ml/metrics.py`, `eval/eval_emotion.py`, `eval/quality_gate.py` |

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
  emotion, as a phase-4 dataset.

`eval_mt.py` scores CTranslate2 on CUDA with INT8/FP16 when a GPU is present, as above; that is
the runtime's `gpu_compute_type`. `--device cpu --compute-type int8` scores the CPU runtime.
The two agree: FLORES chrF 56.72 / BLEU 31.47 on the CPU, against 56.70 / 31.35 on CUDA, with
no leaks on either.

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
  plan moves valence and dominance, but not yet arousal.
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

## Setup

The metrics, the gate and the manifest tools use only the standard library:

```bash
python -m unittest discover -s ml/tests
```

Training, export and full evaluation need `pip install -r ml/requirements.txt`. Check dataset
and model licenses before shipping. Several emotion corpora are research-only, and NLLB-200 is
CC-BY-NC-4.0.
