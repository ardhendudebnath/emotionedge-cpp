# models/

Model files are **not** committed. `manifest.json` lists each one with its task, format, license,
source, size and SHA-256. The runtime's model registry verifies a model before loading it and
refuses a mismatched file.

| id | task | size | license |
|---|---|---|---|
| `vad.silero.v6` | voice activity (1.3) | ~2 MB | MIT |
| `asr.whisper.base.q5_1` | streaming ASR (2.1) | 57 MB | MIT |
| `mt.nllb200.distilled600m.int8` | translation (3.2) | ~620 MB | **CC-BY-NC-4.0, non-commercial** |
| `tts.piper.hi_IN.pratham.medium` | TTS (4.2), Hindi | 61 MB | see the voice's MODEL_CARD |
| `tts.piper.en_US.lessac.medium` | TTS (4.2), English | 60 MB | see the voice's MODEL_CARD |

Check each model's license before shipping. The blueprint notes that several emotion datasets
and some TTS weights are research-only.

## Fetch

```bash
python models/fetch_models.py --list          # what is present / missing
python models/fetch_models.py --pair en-hi    # download + verify everything en->hi needs
python models/fetch_models.py --pin           # pin hashes of entries that have none yet
python models/fetch_models.py --dest ~/ee-models --pair en-hi   # elsewhere; then --set pipeline.models=$HOME/ee-models/manifest.json
```

NLLB-200 is converted rather than downloaded. `ml/export/convert_nllb_ct2.sh` turns
`facebook/nllb-200-distilled-600M` into an INT8 CTranslate2 directory under
`models/mt/nllb-200-distilled-600M-int8/`, including `sentencepiece.bpe.model`.

The en-hi language pack uses the phase-2 emotion-token LoRA of that model,
`mt.nllb200.distilled600m.emo.int8`. It is produced by the recipe in `ml/README.md`
("NLLB emotion-token LoRA"). With plain NLLB instead, run:
`--set translate.model_id=mt.nllb200.distilled600m.int8 --set translate.control_tokens=auto`.

The phase-2 emotion models are exported from Hugging Face. Each export checks ONNX Runtime
against the original model before writing anything, and keeps INT8 only where it does not
change predictions.

```bash
python ml/export/export_emotion2vec_onnx.py --out models/emotion/emotion2vec-plus-base --check data/ravdess.jsonl
python ml/export/export_lexical_onnx.py --out models/emotion/distilroberta-emotion-en --int8
```

- **emotion2vec+ base** stays FP32 (373 MB). INT8 flipped 2 of 64 check clips.
- **DistilRoBERTa** is INT8 (83 MB, 1.9 ms per sentence).
- **Licenses:** emotion2vec+ is under FunASR's model license (attribution). The DistilRoBERTa
  model card states no license, so clear it before shipping.

The phase-3 voice models are exported the same way, each checked against its original.

```bash
python ml/export/export_kokoro_onnx.py --out models/tts/kokoro-82m-hi --espeak-lib <libespeak-ng.so.1>
python ml/export/export_ecapa_onnx.py --out models/speaker/ecapa-voxceleb --check data/ravdess.jsonl
python ml/train/learn_style_offsets.py --corpus data/mt_corpus.hi.jsonl --voice-dir models/tts/kokoro-82m-hi \
    --espeak-lib <libespeak-ng.so.1> --asr-weight 1.0      # optional: per-emotion style offsets (GPU)
```

- **Kokoro-82M** (Apache-2.0) is exported with prosody controls and an exact conv STFT.
  Kokoro's own ONNX STFT is approximate, so it is replaced. With neutral controls, the output
  matches Kokoro's forward pass.
- **ECAPA-TDNN** (SpeechBrain, Apache-2.0) matches SpeechBrain's embeddings with cosine 1.00000.
- **`--espeak-lib`** should point at the espeak-ng the C++ runtime uses, so the golden phonemes
  (`tests/golden/kokoro_g2p_hi.json`) test the C++ port and not an espeak version difference.

## Use

`config/pipeline.engines.yaml` refers to these ids (`model_id: asr.whisper.base.q5_1`). A stage can
also take a plain `model: path/to/file` instead. The build needs the matching engines:

```bash
cmake --preset engines   # EE_WITH_ONNXRUNTIME / WHISPER / CTRANSLATE2 / PIPER
```
