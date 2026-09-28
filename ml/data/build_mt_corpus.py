#!/usr/bin/env python3
"""P1 data for the NLLB emotion-token LoRA (P2): English utterances with an emotion, arousal and
register, written as the exact control prefix the C++ translate stage emits.

Sources: MELD train/dev (TV dialogue, GPL-3.0) and GoEmotions (Reddit, Apache-2.0, single-label
items grouped to Ekman emotions by the dataset's own mapping). MELD test is left alone: it is the
lexical-emotion evaluation set.

    python ml/data/build_mt_corpus.py --meld data/meld_train_sent_emo.csv data/meld_dev_sent_emo.csv \\
        --goemotions data/goemotions_simplified_train.parquet --out data/mt_corpus.jsonl

Each line: {"id", "src", "emotion", "arousal", "register", "prefix", "dataset", "split"}. Hindi
targets are added by ml/data/teacher_translate.py.

How the prefixes are built:
- arousal is the label prototype's arousal plus jitter, quantized to 0.1 as the runtime does;
- a share of items gets the runtime's neutral fallback (`<emo=neutral a=0.0 ...>`), which it
  emits when fused confidence is below τ;
- a few low-arousal neutral/joy items become "calm", the one blueprint label no dataset has.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emotionedge_ml import emotion_space as es  # noqa: E402
from train.finetune_nllb_lora import control_prefix  # noqa: E402

GOEMOTIONS = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity",
              "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear",
              "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization", "relief",
              "remorse", "sadness", "surprise", "neutral"]
# GoEmotions' ekman_mapping.json.
EKMAN = {"anger": ["anger", "annoyance", "disapproval"], "disgust": ["disgust"], "fear": ["fear", "nervousness"],
         "joy": ["joy", "amusement", "approval", "excitement", "gratitude", "love", "optimism", "relief", "pride",
                 "admiration", "desire", "caring"],
         "sadness": ["sadness", "disappointment", "embarrassment", "grief", "remorse"],
         "surprise": ["surprise", "realization", "confusion", "curiosity"], "neutral": ["neutral"]}
TO_EKMAN = {fine: coarse for coarse, fines in EKMAN.items() for fine in fines}

CP1252 = {"\x91": "'", "\x92": "'", "\x93": '"', "\x94": '"', "\x85": "...", "\x96": "-", "\x97": "-",
          "‘": "'", "’": "'", "“": '"', "”": '"', "…": "..."}
BAD = re.compile(r"\[[A-Z]+\]|https?://|www\.|/r/|\bu/|&amp;|&gt;|&lt;")


def clean(text: str) -> str:
    for bad, good in CP1252.items():
        text = text.replace(bad, good)
    return re.sub(r"\s+", " ", text).strip()


def usable(text: str, min_words: int, max_words: int) -> bool:
    words = text.split()
    return min_words <= len(words) <= max_words and not BAD.search(text) and any(c.isalpha() for c in text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--meld", type=Path, nargs="*", default=[])
    parser.add_argument("--goemotions", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--valid", type=int, default=600, help="held-out items")
    parser.add_argument("--neutral-share", type=float, default=0.35, help="cap on neutral items")
    parser.add_argument("--fallback-share", type=float, default=0.15, help="items with the neutral fallback prefix")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    items, seen = [], set()

    def add(text: str, label: str, dataset: str) -> None:
        key = text.lower()
        if key in seen:
            return
        seen.add(key)
        items.append({"src": text, "label": es.DATASET_LABELS[label], "dataset": dataset})

    for path in args.meld:
        with open(path, encoding="utf-8", errors="replace", newline="") as f:
            for row in csv.DictReader(f):
                text = clean(row["Utterance"])
                if usable(text, 1, 40):
                    add(text, row["Emotion"].strip().lower(), "meld")
    if args.goemotions:
        import pyarrow.parquet as pq  # type: ignore

        table = pq.read_table(args.goemotions).to_pydict()
        for text, labels in zip(table["text"], table["labels"]):
            if len(labels) != 1:
                continue
            text = clean(text)
            if usable(text, 3, 30):
                add(text, TO_EKMAN[GOEMOTIONS[labels[0]]], "goemotions")

    neutral = [i for i in items if i["label"] == "neutral"]
    other = [i for i in items if i["label"] != "neutral"]
    keep = int(args.neutral_share / (1 - args.neutral_share) * len(other))
    rng.shuffle(neutral)
    items = other + neutral[:keep]
    rng.shuffle(items)

    for n, item in enumerate(items):
        label = item.pop("label")
        arousal = es.PROTOTYPES[label][1] + rng.uniform(-0.2, 0.2)
        if label in ("neutral", "joy") and rng.random() < 0.08:
            label, arousal = "calm", rng.uniform(-0.6, -0.2)
        register = "casual" if rng.random() < 0.85 else rng.choice(["formal", "neutral"])
        if rng.random() < args.fallback_share:
            label, arousal = "neutral", 0.0  # make_control_tokens below τ
        arousal = max(-1.0, min(1.0, round(arousal, 1)))
        item.update({"id": f"{item['dataset']}-{n}", "emotion": label, "arousal": arousal, "register": register,
                     "prefix": control_prefix(label, arousal, register),
                     "split": "valid" if n < args.valid else "train"})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items), encoding="utf-8")
    from collections import Counter

    print(f"{len(items)} items ({sum(i['split'] == 'valid' for i in items)} valid)")
    print("emotions:", dict(Counter(i["emotion"] for i in items)))
    print("datasets:", dict(Counter(i["dataset"] for i in items)))
    print("example:", items[0]["prefix"], items[0]["src"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
