# ml/: offline ML factory (Python)

Trains, compresses and exports the models the C++ runtime loads. It is never shipped to the
device (blueprint: "Python exists only in the offline factory").

| Step | Blueprint | Here |
|---|---|---|
| P1 Data | IEMOCAP, MSP-Podcast, CREMA-D, ESD; augment noise/RIR/speed | `data/prepare_manifest.py` |
| P2 Train & fine-tune | emotion2vec head → V·A·D; NLLB LoRA + emotion tokens | `train/train_vad_head.py`, `train/finetune_nllb_lora.py` |
| P3 Compress | distillation, INT8 PTQ/QAT, pruning, ONNX simplify | `distill/quantize_onnx.py` |
| P4 Export | torch.onnx / Optimum, CT2 converter, GGML quantize, sign + write manifest | `export/*` |
| P5 Evaluate | WER · COMET · BLEU · emotion F1 · CCC · ECS · latency | `emotionedge_ml/metrics.py`, `eval/quality_gate.py` |

## Contracts with the C++ core

- **Emotion space.** `config/emotion_space.json` holds the label prototypes and the ECS formula.
  Tests in both languages check that the file matches the compiled values.
- **Acoustic model** (`export/export_acoustic_onnx.py`). Input `waveform` [1, samples] at 16 kHz;
  outputs `vad` [1, 3] and `confidence` [1, 3]. This is what `core/emotion/onnx_acoustic.cpp` expects.
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

## Setup

The metrics, the gate and the manifest tools use only the standard library:

```bash
python -m unittest discover -s ml/tests
```

Training, export and full evaluation need `pip install -r ml/requirements.txt`. Check dataset
and model licenses before shipping. Several emotion corpora are research-only, and NLLB-200 is
CC-BY-NC-4.0.
