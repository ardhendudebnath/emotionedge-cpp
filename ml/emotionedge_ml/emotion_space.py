"""The shared V·A·D emotion space (blueprint p.3), read from config/emotion_space.json.

The C++ core compiles the same prototypes (core/emotion/emotion_types.hpp); tests on both sides
check they match, so labels and ECS mean the same thing in training and at runtime.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

REPO = Path(__file__).resolve().parents[2]
SPACE_FILE = REPO / "config" / "emotion_space.json"

Vad = Tuple[float, float, float]

_space = json.loads(SPACE_FILE.read_text(encoding="utf-8"))
PROTOTYPES: Dict[str, Vad] = {k: tuple(v) for k, v in _space["prototypes"].items()}  # type: ignore[misc]
ECS_TARGET: float = float(_space["ecs_target"])
VAD_DIAMETER = 2.0 * math.sqrt(3.0)

# Categorical dataset labels mapped onto the blueprint's label set, for datasets that only have
# categories (CREMA-D, ESD) or when a V·A·D head is trained from categories.
DATASET_LABELS: Dict[str, str] = {
    "neutral": "neutral", "neu": "neutral", "n": "neutral",
    "happy": "joy", "hap": "joy", "joy": "joy", "excited": "joy", "exc": "joy", "h": "joy",
    "surprise": "surprise", "sur": "surprise", "surprised": "surprise",
    "angry": "anger", "ang": "anger", "anger": "anger", "a": "anger", "frustrated": "anger", "fru": "anger",
    "fear": "fear", "fea": "fear", "fearful": "fear", "f": "fear",
    "sad": "sadness", "sadness": "sadness", "s": "sadness",
    "calm": "calm", "relaxed": "calm",
    "disgust": "anger", "dis": "anger", "d": "anger",  # nearest blueprint label on V×A with D+
}


# Classifier class names (emotion2vec+, DistilRoBERTa) -> a position in the space. The blueprint
# labels use their prototypes; disgust, which the blueprint lacks, takes Mehrabian & Russell's
# (1977) PAD rating for "disgusted" rather than being folded into anger.
CLASS_ALIASES: Dict[str, str] = {
    "angry": "anger", "happy": "joy", "sad": "sadness", "fearful": "fear", "surprised": "surprise",
    "disgusted": "disgust",
}
EXTRA_POSITIONS: Dict[str, Vad] = {"disgust": (-0.60, 0.35, 0.11)}
ABSTAIN_CLASSES = ("other", "unknown", "<unk>", "unk")


def class_position(name: str) -> Vad:
    """V·A·D position of a classifier class; raises KeyError for classes without one."""
    key = CLASS_ALIASES.get(name.strip().lower(), name.strip().lower())
    return PROTOTYPES[key] if key in PROTOTYPES else EXTRA_POSITIONS[key]


def label_map(labels: Sequence[str], reliability: Sequence[float] = (1.0, 1.0, 1.0), **extra) -> dict:
    """The `labels.json` a classifier ships with, read by core/emotion/class_mapping.cpp.

    Classes in ABSTAIN_CLASSES carry no emotion evidence; every other class needs a position.
    `extra` adds model-specific fields (e.g. languages, max_tokens)."""
    abstain = [name for name in labels if name.strip().lower() in ABSTAIN_CLASSES]
    positions = {name: list(class_position(name)) for name in labels if name not in abstain}
    neutral = next((name for name in labels if name.strip().lower() == "neutral"), "")
    return {"labels": list(labels), "vad": positions, "abstain": abstain, "neutral": neutral,
            "reliability": list(reliability), **extra}


def distance(x: Sequence[float], y: Sequence[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(x, y)))


def emotion_consistency(src: Sequence[float], out: Sequence[float]) -> float:
    """ECS = 1 - ||VAD_src - VAD_out|| / (2·sqrt(3)), clamped to [0, 1] (blueprint 5.2)."""
    return max(0.0, min(1.0, 1.0 - distance(src, out) / VAD_DIAMETER))


def nearest_label(vad: Sequence[float]) -> str:
    return min(PROTOTYPES, key=lambda label: distance(vad, PROTOTYPES[label]))


def label_to_vad(label: str) -> Vad:
    """V·A·D prototype for a dataset label (case-insensitive); raises KeyError if unknown."""
    return PROTOTYPES[DATASET_LABELS[label.strip().lower()]]


def mean_ecs(pairs: Iterable[Tuple[Sequence[float], Sequence[float]]]) -> float:
    scores = [emotion_consistency(s, o) for s, o in pairs]
    return sum(scores) / len(scores) if scores else 0.0
