#!/usr/bin/env python3
"""P5 "emotion out" evaluation: does the translated speech carry the input's emotion?

1. Build one input WAV from labelled RAVDESS clips. The actors speak neutral sentences in 8
   emotions, so the emotion is in the voice alone:

    python ml/eval/eval_ecs.py build --items data/ravdess.jsonl --out data/ravdess_ecs.wav \\
        --actors 1 2 3 4 --intensity strong

2. Run the pipeline on it (offline) with each TTS configuration:

    emotionedge run --input data/ravdess_ecs.wav --config config/pipeline.engines.yaml --out out/ecs-kokoro

3. Score each run:

    python ml/eval/eval_ecs.py score --clips data/ravdess_ecs.json --session out/ecs-kokoro/session.json --name kokoro

ECS (blueprint 5.2) is 1 − ‖ΔVAD‖ / 2√3 between the input emotion and the emotion that the
consistency stage measures on the synthesized clause. Both sides use the same acoustic model
(emotion2vec+ in pipeline.engines.yaml). Reported per gold emotion and overall, with the share
of clauses at or above the 0.75 target.

Standard library only for `score`. `build` needs numpy + soundfile.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emotionedge_ml import emotion_space as es  # noqa: E402


def build(args) -> int:
    import numpy as np  # type: ignore
    import soundfile as sf  # type: ignore

    rows = [json.loads(line) for line in args.items.read_text(encoding="utf-8").splitlines() if line.strip()]
    # One sentence, the chosen intensity (neutral exists only at normal intensity).
    rows = [r for r in rows if r["actor"] in args.actors and r["text"].startswith("Kids")
            and (r["label"] == "neutral" or r["intensity"] == args.intensity)]
    rows.sort(key=lambda r: (r["actor"], r["label"], r["id"]))
    seen, chosen = set(), []
    for r in rows:  # one clip per actor and emotion
        if (r["actor"], r["label"]) not in seen:
            seen.add((r["actor"], r["label"]))
            chosen.append(r)
    gap = np.zeros(int(args.gap * 16000), dtype=np.float32)
    audio, clips, t = [gap], [], len(gap) / 16000
    for r in chosen:
        x, rate = sf.read(r["audio"], dtype="float32")
        if x.ndim == 2:
            x = x.mean(axis=1)
        if rate != 16000:
            idx = np.arange(0, len(x), rate / 16000)
            x = np.interp(idx, np.arange(len(x)), x).astype(np.float32)
        clips.append({"id": r["id"], "label": r["label"], "actor": r["actor"], "start": t, "end": t + len(x) / 16000})
        audio += [x, gap]
        t += (len(x) + len(gap)) / 16000
    sf.write(args.out, np.concatenate(audio), 16000)
    args.out.with_suffix(".json").write_text(json.dumps(clips, indent=1) + "\n", encoding="utf-8")
    print(f"{len(clips)} clips, {t:.0f} s -> {args.out}")
    return 0


def score(args) -> int:
    clips = json.loads(args.clips.read_text(encoding="utf-8"))
    session = json.loads(args.session.read_text(encoding="utf-8"))
    per_label, detected, n_clauses, above = defaultdict(list), [], 0, 0
    same_label, non_neutral, src_arousal, out_arousal = [], [], [], []
    for u in session["utterances"]:
        src = u.get("source", {})
        mid = (src.get("start", 0.0) + src.get("end", 0.0)) / 2
        clip = next((c for c in clips if c["start"] - 0.3 <= mid <= c["end"] + 0.3), None)
        if clip is None or not u.get("ecs"):
            continue
        gold = es.DATASET_LABELS[clip["label"]]
        per_label[gold] += u["ecs"]
        n_clauses += len(u["ecs"])
        above += sum(e >= es.ECS_TARGET for e in u["ecs"])
        detected.append(u["emotion"]["label"] == gold)
        # Transfer: does the synthesized clause carry the input's emotion? ECS alone is lenient:
        # a neutral output still scores 1 - |src| / 2√3.
        for heard in u.get("output_emotion", []):
            same_label.append(heard["label"] == u["emotion"]["label"])
            non_neutral.append(heard["label"] != "neutral")
            src_arousal.append(u["emotion"]["arousal"])
            out_arousal.append(heard["arousal"])
    if not per_label:
        raise SystemExit("no utterance matched a clip")
    everything = [e for v in per_label.values() for e in v]
    metrics = {f"ecs_mean_{args.name}": sum(everything) / len(everything),
               f"ecs_share_above_target_{args.name}": above / n_clauses,
               f"input_emotion_accuracy_{args.name}": sum(detected) / len(detected)}
    if same_label:
        from emotionedge_ml.metrics import ccc  # noqa: E402

        metrics[f"output_label_agreement_{args.name}"] = sum(same_label) / len(same_label)
        metrics[f"output_non_neutral_share_{args.name}"] = sum(non_neutral) / len(non_neutral)
        metrics[f"arousal_ccc_in_out_{args.name}"] = ccc(src_arousal, out_arousal)
    for label, values in sorted(per_label.items()):
        metrics[f"ecs_{label}_{args.name}"] = sum(values) / len(values)
    print(f"{len(detected)} utterances, {n_clauses} clauses ({args.name})")
    for k, v in metrics.items():
        print(f"  {k:40s} {v:.3f}")
    if args.json:
        existing = json.loads(args.json.read_text(encoding="utf-8")) if args.json.exists() else {}
        existing.update(metrics)
        args.json.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("build")
    p.add_argument("--items", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--actors", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--intensity", default="strong", choices=["strong", "normal"])
    p.add_argument("--gap", type=float, default=1.6, help="seconds of silence between clips")
    p = sub.add_parser("score")
    p.add_argument("--clips", type=Path, required=True)
    p.add_argument("--session", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--json", type=Path)
    args = parser.parse_args()
    return build(args) if args.command == "build" else score(args)


if __name__ == "__main__":
    raise SystemExit(main())
