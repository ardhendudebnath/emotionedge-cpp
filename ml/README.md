# ml/: offline ML factory (Python)

Trains, compresses and exports the models the C++ runtime loads. It is never shipped to the
device (blueprint: "Python exists only in the offline factory").

| Step | Blueprint | Here |
|---|---|---|
| P1 Data | IEMOCAP, MSP-Podcast, CREMA-D, ESD; augment noise/RIR/speed | `data/prepare_manifest.py` |
| P2 Train & fine-tune | emotion2vec head → V·A·D; NLLB LoRA + emotion tokens | `train/train_vad_head.py`, `train/finetune_nllb_lora.py` |
| P3 Compress | distillation, INT8 PTQ/QAT, pruning, ONNX simplify | `distill/quantize_onnx.py` |
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
  `--set translate.control_tokens=on --set translate.arousal_step=0.1`.
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

## Setup

The metrics, the gate and the manifest tools use only the standard library:

```bash
python -m unittest discover -s ml/tests
```

Training, export and full evaluation need `pip install -r ml/requirements.txt`. Check dataset
and model licenses before shipping. Several emotion corpora are research-only, and NLLB-200 is
CC-BY-NC-4.0.
