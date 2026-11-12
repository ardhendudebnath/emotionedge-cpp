#!/usr/bin/env python3
"""How far to trust each automatic judge of emotion out (ml/emotionedge_ml/judges.py).

`ravdess` runs every judge on RAVDESS's neutral, happy, sad and angry speech: acted English, 24
actors, a corpus none of the judges trained on. Each clip is judged as recorded, and trimmed the
way the runtime trims Kokoro's renders (`played()`), since that is how the TTS output gets judged.
Two measures:
- **Four-way:** accuracy, balanced accuracy, recall per class, the confusion matrix and the mean
  top probability. emotion2vec+ is scored on the four classes (its probabilities renormalised
  over them), and audeering's V·A·D by the nearest prototype (config/emotion_space.json).
- **Pairs:** the question the TTS evaluations ask. Each actor says each sentence plainly and with
  the emotion. For every emotional clip and the neutral clip of the same actor, sentence and
  repetition, does the judge hear the emotional one as more of that emotion? For a categorical
  judge that means a higher P(emotion); for audeering, a higher V·A·D projection onto the
  emotion's prototype. Ties count half, and 0.5 is chance.

    python ml/eval/eval_judges.py ravdess --ravdess ~/ee-data/ravdess --json out.json

Requires: torch, torchaudio, soundfile, transformers, funasr.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
from emotionedge_ml import emotion_space as es  # noqa: E402

FOUR = ["neutral", "anger", "joy", "sadness"]
EMOTIONS = ["anger", "joy", "sadness"]
RAVDESS_CODES = {"01": "neutral", "03": "joy", "04": "sadness", "05": "anger"}
INTENSITY = {"01": "normal", "02": "strong"}
JUDGES = ["emotion2vec", "audeering", "iemocap"]
CUTS = ["recorded", "trimmed"]


def load16(path: Path):
    import soundfile as sf  # type: ignore
    import torch  # type: ignore
    import torchaudio  # type: ignore

    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = torch.from_numpy(x.mean(axis=1))
    return torchaudio.functional.resample(x, sr, 16000) if sr != 16000 else x


def four_way(probs: dict) -> dict:
    """Renormalised over the four classes the categorical judges share."""
    total = sum(probs.get(c, 0.0) for c in FOUR) or 1.0
    return {c: probs.get(c, 0.0) / total for c in FOUR}


def along(vad, emotion: str) -> float:
    """V·A·D projected onto the emotion's prototype direction (neutral is the origin)."""
    proto = es.PROTOTYPES[emotion]
    return sum(v * p for v, p in zip(vad, proto)) / math.sqrt(sum(p * p for p in proto))


def nearest(vad) -> str:
    return min(FOUR, key=lambda c: sum((v - p) ** 2 for v, p in zip(vad, es.PROTOTYPES[c])))


def strength(verdict, emotion: str) -> float:
    """How much of the emotion a judge hears: P(emotion), or the projection for V·A·D."""
    return verdict[emotion] if isinstance(verdict, dict) else along(verdict, emotion)


def pair_rates(names, verdict) -> dict:
    """RAVDESS file names and a judge's verdict per name -> the share of pairs in which it hears the
    emotional take as more of the emotion than the neutral take of the same actor, sentence and
    repetition: overall ("all"), per emotion and per emotion/intensity. Ties count half."""
    neutral = {}
    for name in names:
        _, _, emo, _, stmt, rep, actor = Path(name).stem.split("-")
        if RAVDESS_CODES.get(emo) == "neutral":
            neutral[(actor, stmt, rep)] = name
    wins = {}
    for name in names:
        _, _, emo, inten, stmt, rep, actor = Path(name).stem.split("-")
        emotion = RAVDESS_CODES.get(emo)
        if emotion in (None, "neutral") or (actor, stmt, rep) not in neutral:
            continue
        a = strength(verdict(name), emotion)
        b = strength(verdict(neutral[(actor, stmt, rep)]), emotion)
        w = 0.5 if abs(a - b) < 1e-9 else float(a > b)
        for key in ("all", emotion, f"{emotion}/{INTENSITY[inten]}"):
            wins.setdefault(key, []).append(w)
    return {"n": len(wins.get("all", [])), **{k: sum(v) / len(v) for k, v in wins.items()}}


def four_way_scores(rows):
    """rows: (truth, guess, top probability or None)."""
    confusion = {t: {p: 0 for p in FOUR} for t in FOUR}
    for truth, guess, _ in rows:
        confusion[truth][guess] += 1
    recall = {t: confusion[t][t] / sum(confusion[t].values()) for t in FOUR if sum(confusion[t].values())}
    tops = [p for _, _, p in rows if p is not None]
    return {"n": len(rows), "accuracy": sum(confusion[t][t] for t in FOUR) / max(1, len(rows)),
            "balanced_accuracy": sum(recall.values()) / max(1, len(recall)), "recall": recall, "confusion": confusion,
            "mean_top_probability": sum(tops) / len(tops) if tops else None}


def ravdess(args) -> dict:
    import torch  # type: ignore

    from emotionedge_ml.judges import AudeeringJudge, Emotion2vecJudge, IemocapJudge
    from learn_style_offsets import played

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    clips = sorted(p for p in args.ravdess.glob("Actor_*/03-01-*.wav") if p.name.split("-")[2] in RAVDESS_CODES)
    judges = {"emotion2vec": Emotion2vecJudge(dev), "audeering": AudeeringJudge(dev), "iemocap": IemocapJudge(dev)}
    verdicts = {}  # (clip name, cut, judge) -> probs over FOUR, or V·A·D
    for i, path in enumerate(clips):
        audio = load16(path)
        for cut, x in (("recorded", audio), ("trimmed", played(audio, final=True, rate=16000))):
            x = x.numpy()
            verdicts[(path.name, cut, "emotion2vec")] = four_way(judges["emotion2vec"].probs(x))
            verdicts[(path.name, cut, "audeering")] = judges["audeering"].vad(x)
            verdicts[(path.name, cut, "iemocap")] = judges["iemocap"].probs(x)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(clips)}", flush=True)

    out = {"clips": len(clips), "four_way": {}, "pairs": {}, "audeering_mean_vad": {},
           "verdicts": {f"{name}|{cut}|{judge}": v for (name, cut, judge), v in verdicts.items()}}
    print(f"\n{len(clips)} clips. audeering's mean V·A·D, trimmed:")
    for emo, emotion in RAVDESS_CODES.items():
        for inten, level in INTENSITY.items():
            vads = [verdicts[(p.name, "trimmed", "audeering")] for p in clips if p.name.split("-")[2:4] == [emo, inten]]
            if vads:
                mean = [sum(v[k] for v in vads) / len(vads) for k in range(3)]
                out["audeering_mean_vad"][f"{emotion}/{level}"] = mean
                print(f"  {emotion:8s} {level:6s} n {len(vads):3d}  V {mean[0]:+.2f} A {mean[1]:+.2f} D {mean[2]:+.2f}")
    print("Four-way (audeering: nearest prototype):")
    for judge in JUDGES:
        for cut in CUTS:
            rows = []
            for path in clips:
                v = verdicts[(path.name, cut, judge)]
                truth = RAVDESS_CODES[path.name.split("-")[2]]
                if isinstance(v, dict):
                    guess = max(v, key=v.get)
                    rows.append((truth, guess, v[guess]))
                else:
                    rows.append((truth, nearest(v), None))
            s = four_way_scores(rows)
            out["four_way"][f"{judge}/{cut}"] = s
            rec = " ".join(f"{c} {r:.2f}" for c, r in s["recall"].items())
            top = f"  mean top P {s['mean_top_probability']:.2f}" if s["mean_top_probability"] is not None else ""
            print(f"  {judge:11s} {cut:8s} accuracy {s['accuracy']:.3f}  balanced {s['balanced_accuracy']:.3f}  "
                  f"recall: {rec}{top}")
            for t in FOUR:
                print(f"{'':24s}{t:8s} -> " + " ".join(f"{p} {s['confusion'][t][p]:3d}" for p in FOUR))

    print("\nPairs: share of emotional clips heard as more of the emotion than the same actor's neutral take "
          "(0.5 = chance):")
    names = [p.name for p in clips]
    for cut in CUTS:
        for judge in JUDGES:
            rates = pair_rates(names, lambda name: verdicts[(name, cut, judge)])
            out["pairs"][f"{judge}/{cut}"] = rates
            print(f"  {cut:8s} {judge:11s} all {rates['all']:.3f} (n {rates['n']}) | " + " | ".join(
                f"{e} {rates[e]:.2f} (normal {rates[e + '/normal']:.2f}, strong {rates[e + '/strong']:.2f})"
                for e in EMOTIONS))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("ravdess", help="calibrate the judges on RAVDESS")
    p.add_argument("--ravdess", type=Path, required=True, help="directory with Actor_01 ... Actor_24")
    p.add_argument("--json", type=Path)
    args = parser.parse_args()
    result = ravdess(args)
    if args.json:
        args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
