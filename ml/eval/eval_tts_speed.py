#!/usr/bin/env python3
"""How fast can Kokoro speak Hindi and stay intelligible? Sets the cap of the TTS stage's
adaptive pacing (`pacing.max_speed` in config/pipeline.engines.yaml).

    python ml/eval/eval_tts_speed.py --corpus data/mt_corpus.hi.jsonl \\
        --espeak-lib <libespeak-ng.so.1> --speeds 1.0 1.1 1.2 1.3 1.4 1.5

Renders held-out Hindi sentences (the same held-out slice as eval_style_offsets.py) with every
pack voice at each speed. Whisper-small transcribes them. The table gives, per speed:
- the median character error rate (CER);
- the mean CER with each item capped at 1;
- the share of renders Whisper gets badly wrong (CER > 0.5);
- the duration relative to speed 1.0.
Whisper-small's Hindi is weak and it sometimes hallucinates (CER >> 1), so a plain mean is
dominated by a few items.

Requires: torch, torchaudio, kokoro, misaki, transformers.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_style_offsets import cer  # noqa: E402
from learn_style_offsets import EMOJI, REPO, VOICES  # noqa: E402


def main() -> int:
    import torch  # type: ignore
    import torchaudio  # type: ignore

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--espeak-lib", type=Path)
    parser.add_argument("--sentences", type=int, default=24)
    parser.add_argument("--speeds", type=float, nargs="+", default=[1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    from misaki import espeak as misaki_espeak  # type: ignore
    from phonemizer.backend.espeak.wrapper import EspeakWrapper  # type: ignore

    if args.espeak_lib:
        EspeakWrapper.set_library(str(args.espeak_lib))
        EspeakWrapper.set_data_path(str(args.espeak_lib.parent / "espeak-ng-data"))
    g2p = misaki_espeak.EspeakG2P(language="hi")

    rows = [json.loads(line) for line in args.corpus.read_text(encoding="utf-8").splitlines() if line.strip()]
    texts = sorted({r["tgt"] for r in rows if r.get("tgt") and 6 <= len(r["tgt"].split()) <= 16})
    random.Random(5).shuffle(texts)
    held = [(t, g2p(t)[0]) for t in texts[400:400 + args.sentences * 3] if not EMOJI.search(t)]
    held = [(t, p) for t, p in held if 20 <= len(p) <= 200][:args.sentences]

    from huggingface_hub import hf_hub_download  # type: ignore
    from kokoro import KModel  # type: ignore
    from transformers import pipeline  # type: ignore

    kmodel = KModel(repo_id=REPO).to(dev).eval()
    packs = {v: torch.load(hf_hub_download(REPO, f"voices/{v}.pt"), weights_only=True).squeeze(1).float().to(dev)
             for v in VOICES}
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-small", device=0 if dev == "cuda" else -1)

    def render(ps, voice, speed):
        ids = torch.tensor([[0, *[kmodel.vocab[c] for c in ps if c in kmodel.vocab], 0]], device=dev)
        with torch.no_grad():
            audio, _ = kmodel.forward_with_tokens(ids, packs[voice][len(ps) - 1].unsqueeze(0), speed)
        return torchaudio.functional.resample(audio.float().cpu(), 24000, 16000).numpy()

    def transcribe(audio16):
        return asr({"raw": audio16, "sampling_rate": 16000},
                   generate_kwargs={"language": "hindi", "task": "transcribe"})["text"]

    import statistics

    results = {}
    base_seconds = None
    print(f"{len(held)} sentences x {len(VOICES)} voices")
    print(f"{'speed':>6} {'median CER':>11} {'capped mean':>12} {'CER > 0.5':>10} {'duration':>9}")
    for speed in args.speeds:
        errors, seconds = [], 0.0
        for text, ps in held:
            for voice in VOICES:
                audio = render(ps, voice, speed)
                seconds += len(audio) / 16000
                errors.append(cer(text, transcribe(audio)))
        base_seconds = base_seconds or seconds
        row = {"median_cer": statistics.median(errors),
               "capped_mean_cer": sum(min(e, 1.0) for e in errors) / len(errors),
               "bad_share": sum(e > 0.5 for e in errors) / len(errors),
               "duration_ratio": seconds / base_seconds}
        results[f"{speed:.2f}"] = row
        print(f"{speed:6.2f} {row['median_cer']:11.3f} {row['capped_mean_cer']:12.3f} {row['bad_share']:10.0%} "
              f"{row['duration_ratio']:8.2f}x")
    if args.json:
        args.json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
