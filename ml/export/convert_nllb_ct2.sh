#!/usr/bin/env bash
# P4 "CTranslate2 converter (MT)": NLLB-200 -> INT8 CTranslate2 model directory for the C++
# translate stage (core/translate/ct2_translator.cpp), registered in the model manifest.
#
#   ml/export/convert_nllb_ct2.sh [hf-model-or-dir] [output-dir] [model-id] [manifest]
#
# Defaults: facebook/nllb-200-distilled-600M -> models/mt/nllb-200-distilled-600M-int8, registered
# as mt.nllb200.distilled600m.int8 in models/manifest.json. The emotion-token LoRA:
#
#   ml/export/convert_nllb_ct2.sh runs/nllb-emo-lora/merged models/mt/nllb-200-distilled-600M-emo-int8 \
#       mt.nllb200.distilled600m.emo.int8
#
# NLLB-200 weights are CC-BY-NC-4.0 (non-commercial). Requires: pip install ctranslate2 transformers sentencepiece
set -euo pipefail
MODEL=${1:-facebook/nllb-200-distilled-600M}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${2:-$REPO/models/mt/nllb-200-distilled-600M-int8}
ID=${3:-mt.nllb200.distilled600m.int8}
MANIFEST=${4:-$REPO/models/manifest.json}

ct2-transformers-converter --model "$MODEL" --output_dir "$OUT" --quantization int8 --force

# The runtime tokenizes with SentencePiece: ship the model's sentencepiece.bpe.model alongside.
# A fine-tuned model saved with the fast tokenizer may lack it; the vocabulary is the base model's.
python3 - "$MODEL" "$OUT" <<'EOF'
import shutil, sys
from pathlib import Path
from huggingface_hub import hf_hub_download
model, out = sys.argv[1], Path(sys.argv[2])
local = Path(model) / "sentencepiece.bpe.model"
if local.exists():
    src = local
else:
    repo = "facebook/nllb-200-distilled-600M" if Path(model).exists() else model
    src = Path(hf_hub_download(repo, "sentencepiece.bpe.model"))
shutil.copy(src, out / "sentencepiece.bpe.model")
print(f"copied {src} -> {out / 'sentencepiece.bpe.model'}")
EOF

SOURCE="convert:ml/export/convert_nllb_ct2.sh"
[ "$ID" = mt.nllb200.distilled600m.int8 ] || SOURCE="convert:ml/train/finetune_nllb_lora.py + ml/export/convert_nllb_ct2.sh"
python3 "$REPO/ml/export/write_manifest.py" --manifest "$MANIFEST" add --id "$ID" --task mt --format ct2 \
  --path "mt/$(basename "$OUT")" --check-file model.bin --precision int8 --devices cpu cuda \
  --license "CC-BY-NC-4.0 (non-commercial)" --source "$SOURCE"
