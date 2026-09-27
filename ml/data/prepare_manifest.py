#!/usr/bin/env python3
"""P1 "Data": build a JSONL training manifest from emotional-speech corpora, and augment audio.

    python ml/data/prepare_manifest.py --corpus crema-d --root /data/CREMA-D/AudioWAV --out crema.jsonl
    python ml/data/prepare_manifest.py --corpus esd --root /data/ESD --out esd.jsonl
    python ml/data/prepare_manifest.py --corpus csv --root labels.csv --out custom.jsonl

Each line: {"audio": path, "label": blueprint label, "valence": v, "arousal": a, "dominance": d,
"speaker": id, "language": code}. Categorical corpora get the label's V·A·D prototype; corpora with
dimensional ratings (MSP-Podcast, IEMOCAP) should be exported to the `csv` form with their
ratings rescaled to [-1, 1]. Dataset licenses vary and several are research-only: check before
shipping anything trained on them.

Augmentation (noise at an SNR, room impulse response, speed perturbation) lives in `augment()`
and needs numpy + soundfile.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emotionedge_ml.emotion_space import label_to_vad, nearest_label  # noqa: E402

# CREMA-D file names: 1001_DFA_ANG_XX.wav -> emotion code in the third field.
CREMA_CODES = {"ANG": "angry", "DIS": "disgust", "FEA": "fear", "HAP": "happy", "NEU": "neutral", "SAD": "sad"}


def crema_d(root: Path) -> Iterator[Dict]:
    for wav in sorted(root.glob("*.wav")):
        parts = wav.stem.split("_")
        if len(parts) >= 3 and parts[2] in CREMA_CODES:
            yield {"audio": str(wav), "label_raw": CREMA_CODES[parts[2]], "speaker": parts[0], "language": "en"}


def esd(root: Path) -> Iterator[Dict]:
    # ESD layout: <speaker>/<Emotion>/<speaker>_<id>.wav; speakers 0001-0010 Chinese, 0011-0020 English.
    for wav in sorted(root.glob("*/*/*.wav")):
        speaker, emotion = wav.parts[-3], wav.parts[-2]
        language = "zh" if speaker.isdigit() and int(speaker) <= 10 else "en"
        yield {"audio": str(wav), "label_raw": emotion, "speaker": speaker, "language": language}


def from_csv(path: Path) -> Iterator[Dict]:
    # Columns: audio,label[,valence,arousal,dominance,speaker,language]
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            item = {"audio": row["audio"], "label_raw": row.get("label", "neutral"),
                    "speaker": row.get("speaker", ""), "language": row.get("language", "en")}
            if row.get("valence"):
                item["vad"] = [float(row["valence"]), float(row["arousal"]), float(row["dominance"])]
            yield item


def augment(audio, sample_rate: int, noise=None, snr_db: float = 15.0, rir=None, speed: float = 1.0):
    """Returns an augmented copy: speed perturbation, reverberation (RIR), additive noise at an SNR."""
    import numpy as np  # type: ignore

    out = np.asarray(audio, dtype=np.float32)
    if speed != 1.0:
        positions = np.arange(0, len(out), speed)
        out = np.interp(positions, np.arange(len(out)), out).astype(np.float32)
    if rir is not None:
        rir = np.asarray(rir, dtype=np.float32)
        out = np.convolve(out, rir / (np.abs(rir).max() + 1e-9))[: len(out)].astype(np.float32)
    if noise is not None:
        noise = np.resize(np.asarray(noise, dtype=np.float32), len(out))
        signal_power = float(np.mean(out ** 2)) + 1e-12
        noise_power = float(np.mean(noise ** 2)) + 1e-12
        out = out + noise * np.sqrt(signal_power / (noise_power * 10 ** (snr_db / 10)))
    peak = float(np.abs(out).max()) if len(out) else 0.0
    return out / peak * 0.95 if peak > 0.95 else out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", required=True, choices=["crema-d", "esd", "csv"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    readers = {"crema-d": crema_d, "esd": esd, "csv": from_csv}
    count = 0
    skipped = 0
    with args.out.open("w", encoding="utf-8") as out:
        for item in readers[args.corpus](args.root):
            try:
                vad = item.pop("vad", None) or list(label_to_vad(item["label_raw"]))
            except KeyError:
                skipped += 1
                continue
            record = {"audio": item["audio"], "label": nearest_label(vad), "valence": vad[0], "arousal": vad[1],
                      "dominance": vad[2], "speaker": item["speaker"], "language": item["language"]}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    print(f"wrote {count} items to {args.out} ({skipped} with unknown labels skipped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
