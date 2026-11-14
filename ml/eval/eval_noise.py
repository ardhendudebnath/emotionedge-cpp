#!/usr/bin/env python3
"""P5 robustness: what background noise costs the pipeline, before any noise suppression (1.2).

Mixes noise into a RAVDESS input from eval_ecs.py (clips with silence between them) at set
signal-to-noise ratios. It runs the pipeline on each mix and scores it against the clips:

    python ml/eval/eval_ecs.py build --items data/ravdess.jsonl --out data/ravdess_all.wav --actors $(seq 1 24)
    python ml/eval/eval_noise.py run --wav data/ravdess_all.wav --items data/ravdess.jsonl \\
        --emotionedge build/engines/apps/emotionedge --config config/pipeline.engines.yaml --out out/noise \\
        [--noise white pink babble] [--snr 20 10 5 0] [--set KEY=VALUE ...]

- **Noise:**
  - white and pink (1/f) noise;
  - babble: eight talkers at once, from RAVDESS takes of the other sentence ("Dogs are sitting
    by the door"), so no talker says the input's words.
  All of it is generated with a fixed seed; nothing is downloaded.
- **SNR:** speech power over the clips against noise power. The noise runs through the gaps
  too, as a room's would.
- **Scores, per clip:**
  - missed: no utterance in it;
  - extra: utterances outside every clip;
  - WER of the clips' transcripts;
  - the input emotion: is the fused label right? It is scored over all clips, so a missed clip
    counts as wrong.

`mix` writes one noisy input; `score` scores one session. Both `mix` and `run` need numpy and
soundfile; `score` needs only the standard library.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emotionedge_ml import emotion_space as es  # noqa: E402
from emotionedge_ml.metrics import wer  # noqa: E402

RATE = 16000
STATEMENTS = {"01": "Kids are talking by the door.", "02": "Dogs are sitting by the door."}
SLACK_S = 0.3  # an utterance belongs to a clip if its middle is within the clip, give or take this
BABBLE_TALKERS = 8


# ------------------------------------------------------------------------------------- scoring

def score_session(session: dict, clips: List[dict]) -> Dict[str, float]:
    """Missed clips, extra utterances, WER and input-emotion accuracy of one run."""
    utterances = session["utterances"]

    def middle(u: dict) -> float:
        return 0.5 * (u["source"]["start"] + u["source"]["end"])

    def in_clip(u: dict, c: dict) -> bool:
        return c["start"] - SLACK_S <= middle(u) <= c["end"] + SLACK_S

    refs, hyps, missed, right = [], [], 0, 0
    for c in clips:
        mine = [u for u in utterances if in_clip(u, c)]
        refs.append(STATEMENTS[c["id"].split("-")[4]])
        hyps.append(" ".join(u["source"]["text"] for u in mine))
        if not mine:
            missed += 1
            continue
        # The clip's emotion is the one read on its longest utterance.
        longest = max(mine, key=lambda u: u["source"]["end"] - u["source"]["start"])
        right += longest["emotion"]["label"] == es.DATASET_LABELS[c["label"]]
    extra = sum(not any(in_clip(u, c) for c in clips) for u in utterances)
    return {"clips": len(clips), "utterances": len(utterances), "missed": missed, "extra": extra,
            "wer": wer(refs, hyps), "emotion_accuracy": right / len(clips)}


# ------------------------------------------------------------------------------------- mixing

def load16(path: Path):
    import numpy as np  # type: ignore
    import soundfile as sf  # type: ignore

    x, rate = sf.read(str(path), dtype="float32")
    if x.ndim == 2:
        x = x.mean(axis=1)
    if rate != RATE:
        x = np.interp(np.arange(0, len(x), rate / RATE), np.arange(len(x)), x).astype(np.float32)
    return x


def trimmed(x, floor_db: float = -40.0):
    """The take without its leading and trailing silence."""
    import numpy as np  # type: ignore

    loud = np.flatnonzero(np.abs(x) > np.abs(x).max() * 10 ** (floor_db / 20))
    return x[loud[0]:loud[-1] + 1] if len(loud) else x


def make_noise(kind: str, n: int, rng, items: Optional[Path] = None):
    """`n` samples of unit-RMS noise."""
    import numpy as np  # type: ignore

    if kind == "white":
        noise = rng.standard_normal(n)
    elif kind == "pink":
        spectrum = np.fft.rfft(rng.standard_normal(n))
        f = np.fft.rfftfreq(n, 1 / RATE)
        spectrum[1:] /= np.sqrt(f[1:])
        spectrum[0] = 0
        noise = np.fft.irfft(spectrum, n)
    elif kind == "babble":
        if items is None:
            raise SystemExit("babble needs --items (ravdess.jsonl)")
        takes = [json.loads(line) for line in items.read_text(encoding="utf-8").splitlines() if line.strip()]
        takes = [t for t in takes if t["text"].startswith("Dogs")]
        noise = np.zeros(n)
        for _ in range(BABBLE_TALKERS):
            talker, filled = np.zeros(n), 0
            while filled < n:
                x = trimmed(load16(Path(takes[rng.integers(len(takes))]["audio"])))
                x = x / (np.sqrt(np.mean(x ** 2)) + 1e-9)
                take = min(len(x), n - filled)
                talker[filled:filled + take] = x[:take]
                filled += take
            noise += np.roll(talker, rng.integers(n))
    else:
        raise SystemExit(f"unknown noise '{kind}'")
    return (noise / (np.sqrt(np.mean(noise ** 2)) + 1e-12)).astype(np.float32)


def mix(speech, clips: List[dict], noise, snr_db: float):
    """Speech plus noise at `snr_db`, the speech's power taken over the clips."""
    import numpy as np  # type: ignore

    spans = np.zeros(len(speech), dtype=bool)
    for c in clips:
        spans[int(c["start"] * RATE):int(c["end"] * RATE)] = True
    speech_power = float(np.mean(speech[spans] ** 2))
    out = speech + noise * np.sqrt(speech_power / 10 ** (snr_db / 10))
    peak = float(np.abs(out).max())
    return (out / peak * 0.99 if peak > 0.99 else out).astype(np.float32)  # scaling both keeps the SNR


def noisy_input(wav: Path, clips: List[dict], kind: str, snr_db: float, out: Path, items: Optional[Path],
                seed: int) -> Path:
    import numpy as np  # type: ignore
    import soundfile as sf  # type: ignore

    speech = load16(wav)
    noise = make_noise(kind, len(speech), np.random.default_rng(seed), items)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), mix(speech, clips, noise, snr_db), RATE)
    return out


# ------------------------------------------------------------------------------------- commands

def clips_of(wav: Path) -> List[dict]:
    return json.loads(wav.with_suffix(".json").read_text(encoding="utf-8"))


def cmd_mix(args) -> int:
    noisy_input(args.wav, clips_of(args.wav), args.noise, args.snr, args.out, args.items, args.seed)
    print(f"wrote {args.out}")
    return 0


def cmd_score(args) -> int:
    clips = json.loads(args.clips.read_text(encoding="utf-8"))
    print(json.dumps(score_session(json.loads(args.session.read_text(encoding="utf-8")), clips), indent=1))
    return 0


def table(results: Dict[str, dict]) -> str:
    lines = ["| Input | Missed clips | Extra utterances | WER | Emotion read right |", "|---|---|---|---|---|"]
    for name, r in results.items():
        lines.append(f"| {name} | {r['missed']}/{r['clips']} | {r['extra']} | {r['wer']:.3f} | {r['emotion_accuracy']:.3f} |")
    return "\n".join(lines)


def cmd_run(args) -> int:
    clips = clips_of(args.wav)
    conditions = [("clean", None, None)] + [(f"{kind} {snr:g} dB", kind, snr) for kind in args.noise for snr in args.snr]
    path = args.out / "noise.json"
    results = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for name, kind, snr in conditions:
        if name in results and not args.force:
            print(f"{name}: done before, kept")
            continue
        slug = name.replace(" ", "-")
        wav = args.wav if kind is None else noisy_input(args.wav, clips, kind, snr, args.out / "inputs" / f"{slug}.wav",
                                                       args.items, args.seed)
        run = args.out / "runs" / slug
        cmd = [str(args.emotionedge), "run", "--input", str(wav), "--config", str(args.config), "--out", str(run)]
        for kv in args.set:
            cmd += ["--set", kv]
        if args.log:
            cmd += ["--log", args.log]
        print(f"{name}: running", flush=True)
        run.parent.mkdir(parents=True, exist_ok=True)
        with run.with_suffix(".log").open("w") as log:
            subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
        results[name] = score_session(json.loads((run / "session.json").read_text(encoding="utf-8")), clips)
        print(f"{name}: {results[name]}", flush=True)
        path.write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    print(table(results))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run", help="every noise at every SNR, plus the clean input")
    p.add_argument("--wav", type=Path, required=True, help="eval_ecs.py build output (its .json beside it)")
    p.add_argument("--items", type=Path, help="ravdess.jsonl: babble is built from it")
    p.add_argument("--emotionedge", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--noise", nargs="+", default=["white", "pink", "babble"])
    p.add_argument("--snr", type=float, nargs="+", default=[20, 10, 5, 0])
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--force", action="store_true", help="rerun conditions already in noise.json")
    p.add_argument("--log", help="the pipeline's log level (debug: whisper's language detections in runs/*.log)")
    p = sub.add_parser("mix", help="one noisy input")
    p.add_argument("--wav", type=Path, required=True)
    p.add_argument("--noise", required=True, choices=["white", "pink", "babble"])
    p.add_argument("--snr", type=float, required=True)
    p.add_argument("--items", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=1)
    p = sub.add_parser("score", help="one session")
    p.add_argument("--clips", type=Path, required=True)
    p.add_argument("--session", type=Path, required=True)
    args = parser.parse_args(argv)
    return {"run": cmd_run, "mix": cmd_mix, "score": cmd_score}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
