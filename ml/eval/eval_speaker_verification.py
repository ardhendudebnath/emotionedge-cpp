#!/usr/bin/env python3
"""Calibrates speaker-aware barge-in (core/audio/speaker.cpp): when speech starts while a
translation plays, the speaker stage embeds the first `barge_in_ms` of it with ECAPA-TDNN and
compares it with the interpreted speaker's enrolled print. A different voice cancels the
translation. The voice of the TTS itself, picked up by the microphone, must not.

    python ml/eval/eval_speaker_verification.py --manifest data/ravdess.jsonl \\
        --kokoro-corpus data/mt_corpus.hi.jsonl --espeak-lib <libespeak-ng.so.1>

Speakers: the 24 RAVDESS actors (12 male, 12 female), acted in 8 emotions, so the same voice is
compared across emotions.
- Each actor's reference print is the mean of enrolled clips (up to 2 s of speech each), as the
  runtime enrols from earlier utterances. There are two references:
  - cold: 3 neutral/calm clips;
  - warm: 8 clips of any emotion, as after a few utterances.
- Probes are the first T seconds of speech of every clip not enrolled.
- Scores are the cosine against the actor's own reference (target) and against every other
  actor's (non-target).

For each probe length, the table gives the equal error rate, and per threshold:
- false barge-in: the speaker's own speech judged different;
- missed barge-in: another speaker judged the same, also counted for same-gender speakers only.
With --kokoro-corpus, Kokoro's Hindi voices stand in for the echo of the translation: their
probes against their own print (echo recognized) and against the actors' references.

Requires: torch, torchaudio, soundfile, speechbrain, librosa (+ kokoro, misaki for the echo check).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

SOURCE = "speechbrain/spkrec-ecapa-voxceleb"


def main() -> int:
    import librosa  # type: ignore
    import numpy as np  # type: ignore
    import soundfile  # type: ignore
    import torch  # type: ignore
    import torchaudio  # type: ignore
    from speechbrain.inference.speaker import EncoderClassifier  # type: ignore

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True, help="ravdess.jsonl (eval_emotion.py prepare-ravdess)")
    parser.add_argument("--probe-seconds", type=float, nargs="+", default=[0.75, 1.0, 1.5, 2.0])
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4])
    parser.add_argument("--kokoro-corpus", type=Path, help="Hindi texts (mt_corpus.hi.jsonl) for the echo check")
    parser.add_argument("--espeak-lib", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    enc = EncoderClassifier.from_hparams(source=SOURCE, savedir=str(Path.home() / "ee-ml" / "sb-ecapa"),
                                         run_opts={"device": "cpu"})

    def embed(wav16):
        with torch.no_grad():
            e = enc.encode_batch(torch.from_numpy(wav16).float().unsqueeze(0)).squeeze().numpy()
        return e / np.linalg.norm(e)

    def speech(path_or_audio, rate=None):
        """16 kHz mono, trimmed to its speech (the segmenter's job at runtime)."""
        if rate is None:
            data, rate = soundfile.read(str(path_or_audio), dtype="float32", always_2d=True)
            path_or_audio = data.mean(axis=1)
        x = torch.from_numpy(np.ascontiguousarray(path_or_audio)).float()
        x = torchaudio.functional.resample(x, rate, 16000).numpy()
        y, _ = librosa.effects.trim(x, top_db=35)
        return y

    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    clips = defaultdict(list)
    for r in rows:
        clips[r["actor"]].append(r)
    # Every clip once: its print as an enrolled utterance (up to 2 s, the runtime's max_seconds)
    # and as a probe of each length.
    full, probe = {}, {}
    for actor, items in sorted(clips.items()):
        for r in items:
            s = speech(r["audio"])
            full[r["id"]] = embed(s[:32000])
            probe[r["id"]] = {t: embed(s[:int(t * 16000)]) for t in args.probe_seconds}
        print(f"actor {actor:2d}: {len(items)} clips", file=sys.stderr)

    def male(actor):
        return actor % 2 == 1

    # Two references per actor, as at runtime before and after a few utterances:
    # - cold: 3 neutral/calm clips;
    # - warm: 8 clips of any emotion (the runtime refines its reference with every utterance
    #   in the same voice).
    rng = random.Random(3)
    conditions = {}
    for name in ("cold", "warm"):
        refs, enrolled = {}, set()
        for actor, items in sorted(clips.items()):
            items = sorted(items, key=lambda r: r["id"])
            if name == "cold":
                enrol = [r for r in items if r["label"] in ("neutral", "calm") and r.get("intensity") == "normal"][:3]
            else:
                enrol = rng.sample(items, 8)
            ref = np.mean([full[r["id"]] for r in enrol], axis=0)
            refs[actor] = ref / np.linalg.norm(ref)
            enrolled |= {r["id"] for r in enrol}
        conditions[name] = (refs, [(r["actor"], r["id"]) for r in rows if r["id"] not in enrolled])

    results = {}
    for name, (refs, probes) in conditions.items():
        print(f"\n### {name} reference: {len(refs)} speakers, {len(probes)} probes")
        for t in args.probe_seconds:
            target = np.array([float(refs[a] @ probe[i][t]) for a, i in probes])
            nontarget = np.array([float(refs[b] @ probe[i][t]) for a, i in probes for b in refs if b != a])
            same_gender = np.array([float(refs[b] @ probe[i][t]) for a, i in probes for b in refs
                                    if b != a and male(b) == male(a)])
            # Equal error rate: the threshold where false barge-ins equal missed ones.
            grid = np.linspace(-0.2, 1.0, 1201)
            frr = np.array([(target < g).mean() for g in grid])
            far = np.array([(nontarget >= g).mean() for g in grid])
            k = int(np.argmin(np.abs(frr - far)))
            row = {"eer": float((frr[k] + far[k]) / 2), "eer_threshold": float(grid[k]), "thresholds": {}}
            print(f"probe {t:.2f} s: EER {row['eer']:.1%} at {grid[k]:.2f}; "
                  f"target cos median {np.median(target):.2f}, non-target {np.median(nontarget):.2f}")
            print(f"  {'threshold':>9} {'false barge-in':>15} {'missed':>8} {'missed (same gender)':>21}")
            for g in args.thresholds:
                cell = {"false_barge_in": float((target < g).mean()), "missed": float((nontarget >= g).mean()),
                        "missed_same_gender": float((same_gender >= g).mean())}
                row["thresholds"][f"{g:.2f}"] = cell
                print(f"  {g:9.2f} {cell['false_barge_in']:15.1%} {cell['missed']:8.1%} {cell['missed_same_gender']:21.1%}")
            results[f"{name}_{t:.2f}"] = row
    refs = conditions["warm"][0]

    if args.kokoro_corpus:
        # The translation's own voice reaching the microphone: Kokoro probes against the print of
        # that voice's earlier output (echo, to ignore) and against the actors (not a speaker).
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
        from learn_style_offsets import EMOJI, REPO, VOICES  # noqa: E402
        from huggingface_hub import hf_hub_download  # type: ignore
        from kokoro import KModel  # type: ignore
        from misaki import espeak as misaki_espeak  # type: ignore
        from phonemizer.backend.espeak.wrapper import EspeakWrapper  # type: ignore

        if args.espeak_lib:
            EspeakWrapper.set_library(str(args.espeak_lib))
            EspeakWrapper.set_data_path(str(args.espeak_lib.parent / "espeak-ng-data"))
        g2p = misaki_espeak.EspeakG2P(language="hi")
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        kmodel = KModel(repo_id=REPO).to(dev).eval()
        corpus = [json.loads(line) for line in args.kokoro_corpus.read_text(encoding="utf-8").splitlines() if line.strip()]
        texts = sorted({r["tgt"] for r in corpus if r.get("tgt") and 6 <= len(r["tgt"].split()) <= 16})
        random.Random(7).shuffle(texts)
        texts = [t for t in texts[:40] if not EMOJI.search(t)][:9]
        own_scores, actor_scores = defaultdict(list), defaultdict(list)
        for voice in VOICES:
            pack = torch.load(hf_hub_download(REPO, f"voices/{voice}.pt"), weights_only=True).squeeze(1).float().to(dev)
            renders = []
            for text in texts:
                ps, _ = g2p(text)
                ids = torch.tensor([[0, *[kmodel.vocab[c] for c in ps if c in kmodel.vocab], 0]], device=dev)
                with torch.no_grad():
                    audio, _ = kmodel.forward_with_tokens(ids, pack[len(ps) - 1].unsqueeze(0), 1.0)
                renders.append(speech(audio.float().cpu().numpy(), 24000))
            own = np.mean([embed(r[:32000]) for r in renders[:3]], axis=0)
            own /= np.linalg.norm(own)
            for r in renders[3:]:
                for t in args.probe_seconds:
                    p = embed(r[:int(t * 16000)])
                    own_scores[t].append(float(own @ p))
                    actor_scores[t] += [float(refs[a] @ p) for a in refs]
        print("\nKokoro voices as echo of the translation (probes vs the voice's own print / vs the actors):")
        for t in args.probe_seconds:
            o, a = np.array(own_scores[t]), np.array(actor_scores[t])
            print(f"  probe {t:.2f} s: own cos median {np.median(o):.2f} (min {o.min():.2f}); "
                  f"vs actors median {np.median(a):.2f} (max {a.max():.2f})")
            results[f"echo_{t:.2f}"] = {"own_median": float(np.median(o)), "own_min": float(o.min()),
                                        "vs_actors_max": float(a.max())}

    if args.json:
        args.json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
