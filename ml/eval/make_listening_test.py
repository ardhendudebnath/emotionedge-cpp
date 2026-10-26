#!/usr/bin/env python3
"""Clips for a listening test of Kokoro's emotion out, as the runtime renders it.

The automatic judges disagree on whether the shipped settings make Kokoro's Hindi sound angry,
happy or sad (ml/README.md, "Played as at runtime"), so people decide. Each trial pairs two
renders of one sentence in one voice with the same decoder noise:
- neutral: no style offset, Kokoro's own prosody;
- emotional: the emotion's shipped style offset and the controller's plan (the rules, or the
  learned plan for the emotions the runtime lists in plan_emotions), both at the strength the
  runtime applies (0.7 is its RAVDESS median; some trials use 1.0).

Clips are played as at runtime (silence trimmed) and matched in loudness, so the choice rests on
the voice. The sentences say nothing emotional, so the words do not carry the emotion: RAVDESS's
two statements in Hindi (the clips eval/eval_ecs.py runs end to end), and one more of the kind.
Writes <out>/audio/*.wav and <out>/trials.json.

    python ml/eval/make_listening_test.py --voice-dir ~/ee-models/tts/kokoro-82m-hi \\
        --espeak-lib .../libespeak-ng.so.1 --out out/listening

Requires: torch, kokoro, misaki, soundfile.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
from learn_controller import NEUTRAL, KokoroPlayback, rules_at, scaled  # noqa: E402

VOICE_LABELS = {"hf_alpha": "female voice", "hf_beta": "female voice 2", "hm_omega": "male voice", "hm_psi": "male voice 2"}
# Emotionally empty carrier sentences, as emotion-perception studies use: RAVDESS's statements 01
# and 02, and one more of the kind.
SENTENCES = [
    ("बच्चे दरवाज़े के पास बात कर रहे हैं।", "Kids are talking by the door."),
    ("कुत्ते दरवाज़े के पास बैठे हैं।", "Dogs are sitting by the door."),
    ("मैंने किताब मेज़ पर रख दी है।", "I have put the book on the table."),
]


def main() -> int:
    import numpy as np  # type: ignore
    import soundfile as sf  # type: ignore
    import torch  # type: ignore

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--voice-dir", type=Path, required=True)
    parser.add_argument("--espeak-lib", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--emotions", nargs="+", default=["anger", "joy", "sadness"],
                        help="emotions with a shipped style offset (style_emotions)")
    parser.add_argument("--plan-emotions", nargs="+", default=["sadness"],
                        help="emotions the runtime renders with the learned plan (plan_emotions); the rest use the rules")
    parser.add_argument("--voices", nargs="+", default=["hf_alpha", "hm_omega"])
    parser.add_argument("--sentences", type=int, default=len(SENTENCES), help="how many of SENTENCES, at --strength")
    parser.add_argument("--strength", type=float, default=0.7)
    parser.add_argument("--full-strength-sentences", type=int, default=2,
                        help="of those, how many also get a trial at strength 1.0")
    parser.add_argument("--rms-db", type=float, default=-20.0, help="loudness every clip is matched to")
    args = parser.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    from misaki import espeak as misaki_espeak  # type: ignore
    from phonemizer.backend.espeak.wrapper import EspeakWrapper  # type: ignore

    if args.espeak_lib:
        EspeakWrapper.set_library(str(args.espeak_lib))
        EspeakWrapper.set_data_path(str(args.espeak_lib.parent / "espeak-ng-data"))
    g2p = misaki_espeak.EspeakG2P(language="hi")

    kokoro = KokoroPlayback(args.voice_dir / "style_offsets.json", args.emotions, dev)
    plan = json.loads((args.voice_dir / "prosody_controller.json").read_text(encoding="utf-8"))["controls"]

    audio_dir = args.out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    def save(name, audio):
        x = audio.detach().cpu().numpy().astype(np.float64)
        x *= 10 ** (args.rms_db / 20) / max(np.sqrt(np.mean(x ** 2)), 1e-9)
        x *= min(1.0, 10 ** (-1 / 20) / max(np.abs(x).max(), 1e-9))  # peaks at most -1 dBFS
        sf.write(audio_dir / f"{name}.wav", x.astype(np.float32), 24000, subtype="PCM_16")
        return f"audio/{name}.wav"

    trials = []
    for i, (hindi, english) in enumerate(SENTENCES[:args.sentences]):
        ps = g2p(hindi)[0]
        for j, voice in enumerate(args.voices):
            seed = 70000 + 100 * i + j  # the same noise for a pair
            neutral = save(f"s{i}-{voice}-neutral", kokoro.render(ps, voice, None, NEUTRAL, seed))
            strengths = [args.strength] + ([1.0] if i < args.full_strength_sentences and args.strength != 1.0 else [])
            for emotion in args.emotions:
                for k in strengths:
                    c = scaled(plan[emotion], k) if emotion in args.plan_emotions else rules_at(emotion, k)
                    emotional = save(f"s{i}-{voice}-{emotion}-{k:g}", kokoro.render(ps, voice, emotion, c, seed, strength=k))
                    trials.append({"id": f"s{i}-{voice}-{emotion}-{k:g}", "emotion": emotion, "strength": k,
                                   "voice": VOICE_LABELS.get(voice, voice), "sentence": hindi, "english": english,
                                   "plan": "learned" if emotion in args.plan_emotions else "rules",
                                   "neutral": neutral, "emotional": emotional})
    (args.out / "trials.json").write_text(json.dumps({"trials": trials}, ensure_ascii=False, indent=1) + "\n",
                                          encoding="utf-8")
    print(f"{len(trials)} trials, {len(list(audio_dir.glob('*.wav')))} clips in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
