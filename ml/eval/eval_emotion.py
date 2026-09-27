#!/usr/bin/env python3
"""P5 emotion evaluation of the C++ emotion models (blueprint 2.2) on labelled data.

1. Prepare items (JSONL with id, audio and/or text, label):

    python ml/eval/eval_emotion.py prepare-ravdess --zip Audio_Speech_Actors_01-24.zip \\
        --dir data/ravdess --out data/ravdess.jsonl       # acoustic: acted speech, 8 emotions
    python ml/eval/eval_emotion.py prepare-meld --csv test_sent_emo.csv --out data/meld_test.jsonl
                                                         # lexical: TV-dialogue text, 7 emotions

2. Predict with the runtime itself, so the C++ inference is what gets measured:

    emotionedge emotion --config config/pipeline.engines.yaml --manifest data/ravdess.jsonl \\
        --out preds.jsonl [--set emotion.acoustic=onnx ...]

3. Score (the prediction is the blueprint label nearest to the modality's V·A·D point):

    python ml/eval/eval_emotion.py score --items data/ravdess.jsonl --preds preds.jsonl \\
        --modality acoustic [--name acoustic] [--json metrics.json]

Metrics: UAR (unweighted average recall, the usual SER metric), macro-F1, accuracy, and CCC per
axis against the gold label's prototype (a coarse proxy: categorical data has no V·A·D ratings).
Standard library only.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emotionedge_ml import emotion_space as es  # noqa: E402
from emotionedge_ml.metrics import macro_f1, summarize_vad  # noqa: E402

# RAVDESS file names: modality-channel-EMOTION-intensity-statement-repetition-actor.wav
RAVDESS_EMOTIONS = {"01": "neutral", "02": "calm", "03": "happy", "04": "sad", "05": "angry",
                    "06": "fearful", "07": "disgust", "08": "surprised"}
RAVDESS_STATEMENTS = {"01": "Kids are talking by the door.", "02": "Dogs are sitting by the door."}


def prepare_ravdess(args) -> int:
    args.dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip) as z:
        z.extractall(args.dir)
    rows = []
    for wav in sorted(args.dir.rglob("*.wav")):
        parts = wav.stem.split("-")
        if len(parts) != 7 or parts[0] != "03":  # audio-only speech files
            continue
        rows.append({"id": wav.stem, "audio": str(wav.resolve()), "text": RAVDESS_STATEMENTS[parts[4]],
                     "label": RAVDESS_EMOTIONS[parts[2]], "intensity": "strong" if parts[3] == "02" else "normal",
                     "actor": int(parts[6])})
    write_jsonl(args.out, rows)
    print(f"{len(rows)} items, labels {dict(Counter(r['label'] for r in rows))}")
    return 0


def prepare_meld(args) -> int:
    rows = []
    with open(args.csv, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            text = r["Utterance"].replace("\u0092", "'").replace("\u0085", "...").strip()
            rows.append({"id": f"dia{r['Dialogue_ID']}_utt{r['Utterance_ID']}", "text": text, "label": r["Emotion"]})
    write_jsonl(args.out, rows)
    print(f"{len(rows)} items, labels {dict(Counter(r['label'] for r in rows))}")
    return 0


def write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def read_jsonl(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def score(args) -> int:
    items = {r["id"]: r for r in read_jsonl(args.items)}
    truth, pred, gold_vad, pred_vad, skipped = [], [], [], [], 0
    for p in read_jsonl(args.preds):
        item = items.get(p["id"])
        estimate = p.get(args.modality, {})
        if item is None or (args.modality != "fused" and not estimate.get("valid")):
            skipped += 1
            continue
        gold = es.DATASET_LABELS[item["label"].strip().lower()]
        truth.append(gold)
        pred.append(es.nearest_label(estimate["vad"]))
        gold_vad.append(es.PROTOTYPES[gold])
        pred_vad.append(estimate["vad"])
    if not truth:
        raise SystemExit("no scorable predictions")
    labels = sorted(set(truth))
    per_class: Dict[str, float] = {}
    for label in labels:
        hits = [p == t for t, p in zip(truth, pred) if t == label]
        per_class[label] = sum(hits) / len(hits)
    name = args.name or args.modality
    metrics = {
        f"emotion_uar_{name}": sum(per_class.values()) / len(per_class),
        f"emotion_f1_{name}": macro_f1(truth, pred, labels),
        f"emotion_accuracy_{name}": sum(t == p for t, p in zip(truth, pred)) / len(truth),
        **{f"{k}_{name}": v for k, v in summarize_vad(gold_vad, pred_vad).items()},
    }
    print(f"{len(truth)} scored, {skipped} skipped ({args.modality})")
    for k, v in metrics.items():
        print(f"  {k:32s} {v:.3f}")
    print("  recall per class: " + ", ".join(f"{k} {v:.2f}" for k, v in per_class.items()))
    confusion: Dict[str, Counter] = defaultdict(Counter)
    for t, p in zip(truth, pred):
        confusion[t][p] += 1
    print("  confusion (row = gold): " + "; ".join(
        f"{t}: " + ", ".join(f"{p} {n}" for p, n in c.most_common(3)) for t, c in sorted(confusion.items())))
    if args.json:
        existing = json.loads(args.json.read_text(encoding="utf-8")) if args.json.exists() else {}
        existing.update(metrics)
        args.json.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare-ravdess")
    p.add_argument("--zip", type=Path, required=True)
    p.add_argument("--dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("prepare-meld")
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("score")
    p.add_argument("--items", type=Path, required=True)
    p.add_argument("--preds", type=Path, required=True)
    p.add_argument("--modality", choices=["acoustic", "lexical", "fused"], required=True)
    p.add_argument("--name", help="metric suffix (default: the modality)")
    p.add_argument("--json", type=Path, help="merge the metrics into this JSON file")
    args = parser.parse_args()
    return {"prepare-ravdess": prepare_ravdess, "prepare-meld": prepare_meld, "score": score}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
