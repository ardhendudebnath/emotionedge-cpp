#!/usr/bin/env bash
# P4 "GGML quantize (whisper)": re-quantize a whisper.cpp GGML model (e.g. a fine-tuned export)
# with whisper.cpp's quantize tool, then register it in models/manifest.json.
#
#   ml/export/quantize_whisper.sh <input.bin> <output.bin> [q5_1|q8_0|q4_0]
#
# Needs whisper.cpp built with its examples (the `quantize` tool): see github.com/ggml-org/whisper.cpp
set -euo pipefail
IN=$1
OUT=$2
TYPE=${3:-q5_1}
QUANTIZE=${WHISPER_QUANTIZE:-quantize}
REPO=$(cd "$(dirname "$0")/../.." && pwd)

"$QUANTIZE" "$IN" "$OUT" "$TYPE"
REL=$(python3 -c "import os,sys; print(os.path.relpath(sys.argv[1], sys.argv[2]))" "$OUT" "$REPO/models")
python3 "$REPO/ml/export/write_manifest.py" add --id "asr.whisper.custom.$TYPE" --task asr --format ggml \
  --path "$REL" --precision "$TYPE" --devices cpu cuda --license MIT
