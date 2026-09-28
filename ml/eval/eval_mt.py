#!/usr/bin/env python3
"""P5 MT evaluation of CTranslate2 NLLB models, fed the way core/translate/ct2_translator.cpp feeds
them: the control prefix encoded as its own SentencePiece pieces before the text, beam 2.

    python ml/eval/eval_mt.py --model ~/ee-models/mt/nllb-200-distilled-600M-int8 --name vanilla \\
        --flores data/flores200_dataset --corpus data/mt_corpus.hi.jsonl --prefix none \\
        --backtranslate ~/ee-models/mt/nllb-200-distilled-600M-int8 --json mt_metrics.json

--prefix sets what the model is given:
- `none`: plain text (what a vanilla model gets, since the runtime strips the prefix);
- `runtime`: each corpus item's own prefix, with the neutral fallback prefix on FLORES.

Metrics, suffixed with --name:
- chrF / BLEU on FLORES-200 devtest eng->hin (human references: general quality);
- chrF on the corpus's held-out split against the teacher's targets (in-domain);
- the leak rate: outputs containing control-prefix text;
- emotion round trip: each output is back-translated to English with --backtranslate, then
  DistilRoBERTa classifies both the source and the round trip. The metrics are the share of
  labels that agree, and the mean text ECS (1 − ‖ΔVAD‖/2√3) between the two.

Requires: ctranslate2, sentencepiece, sacrebleu; transformers + torch for the round trip.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emotionedge_ml import emotion_space as es  # noqa: E402

NEUTRAL_PREFIX = "<emo=neutral a=0.0 reg=casual>"
CLASSIFIER = "j-hartmann/emotion-english-distilroberta-base"


class Ct2Nllb:
    def __init__(self, model_dir: Path, beam: int = 2, device: str = "auto", compute_type: str = "auto") -> None:
        import ctranslate2  # type: ignore
        import sentencepiece as spm  # type: ignore

        # The runtime's two settings: CPU int8, or CUDA int8_float16 (pipeline.engines.yaml).
        if device == "auto":
            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        if compute_type == "auto":
            compute_type = "int8_float16" if device == "cuda" else "int8"
        self.translator = ctranslate2.Translator(str(model_dir), device=device, compute_type=compute_type)
        self.sp = spm.SentencePieceProcessor(model_file=str(model_dir / "sentencepiece.bpe.model"))
        self.beam = beam

    def translate(self, sources, prefixes, src="eng_Latn", tgt="hin_Deva", batch=32):
        tokens = [[src] + (self.sp.encode(p, out_type=str) if p else []) + self.sp.encode(s, out_type=str) + ["</s>"]
                  for s, p in zip(sources, prefixes)]
        results = self.translator.translate_batch(tokens, target_prefix=[[tgt]] * len(tokens), beam_size=self.beam,
                                                  max_batch_size=batch, max_decoding_length=256)
        return [self.sp.decode(r.hypotheses[0][1:]) for r in results]


def leaks(text: str) -> bool:
    return any(marker in text for marker in ("<emo", "emo=", "reg=", "a=0.", "a=-0.", "a=1.", "casual>"))


def main() -> int:
    import sacrebleu  # type: ignore

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--prefix", choices=["none", "runtime"], required=True)
    parser.add_argument("--flores", type=Path, help="flores200_dataset directory")
    parser.add_argument("--corpus", type=Path, help="JSONL with prefix/src/tgt/split (valid split is used)")
    parser.add_argument("--backtranslate", type=Path, help="CT2 NLLB model for hin->eng round trips")
    parser.add_argument("--device", default="auto", help="cpu | cuda | auto (cuda when available)")
    parser.add_argument("--compute-type", default="auto", help="auto: int8 on cpu, int8_float16 on cuda")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    mt = Ct2Nllb(args.model, device=args.device, compute_type=args.compute_type)
    metrics = {}
    suffix = f"_{args.name}"
    if args.flores:
        src = (args.flores / "devtest" / "eng_Latn.devtest").read_text(encoding="utf-8").splitlines()
        ref = (args.flores / "devtest" / "hin_Deva.devtest").read_text(encoding="utf-8").splitlines()
        prefixes = [NEUTRAL_PREFIX if args.prefix == "runtime" else ""] * len(src)
        hyp = mt.translate(src, prefixes)
        metrics["chrf_flores" + suffix] = sacrebleu.corpus_chrf(hyp, [ref]).score
        metrics["bleu_flores" + suffix] = sacrebleu.corpus_bleu(hyp, [ref]).score
        metrics["leak_rate_flores" + suffix] = sum(map(leaks, hyp)) / len(hyp)

    if args.corpus:
        rows = [json.loads(line) for line in args.corpus.read_text(encoding="utf-8").splitlines() if line.strip()]
        valid = [r for r in rows if r.get("split") == "valid" and r.get("tgt")]
        hyp = mt.translate([r["src"] for r in valid], [r["prefix"] if args.prefix == "runtime" else "" for r in valid])
        metrics["chrf_indomain" + suffix] = sacrebleu.corpus_chrf(hyp, [[r["tgt"] for r in valid]]).score
        metrics["leak_rate_indomain" + suffix] = sum(map(leaks, hyp)) / len(hyp)
        for r, h in list(zip(valid, hyp))[:6]:
            print(f"  {r['prefix'] if args.prefix == 'runtime' else '':34s} {r['src'][:60]!r}\n    -> {h}")

        if args.backtranslate:
            from transformers import pipeline  # type: ignore

            back = Ct2Nllb(args.backtranslate).translate(hyp, [""] * len(hyp), src="hin_Deva", tgt="eng_Latn")
            classify = pipeline("text-classification", model=CLASSIFIER, top_k=None, device=0 if _cuda() else -1)
            labels = [s["label"] for s in classify([r["src"] for r in valid][:1])[0]]
            positions = {lb: es.class_position(lb) for lb in labels}

            def vad(scores):
                return tuple(sum(s["score"] * positions[s["label"]][k] for s in scores) for k in range(3))

            def top(scores):
                return es.DATASET_LABELS[max(scores, key=lambda s: s["score"])["label"]]

            s_scores = classify([r["src"] for r in valid], batch_size=64, truncation=True)
            b_scores = classify(back, batch_size=64, truncation=True)
            metrics["emotion_roundtrip_agreement" + suffix] = sum(
                top(a) == top(b) for a, b in zip(s_scores, b_scores)) / len(valid)
            metrics["emotion_roundtrip_ecs" + suffix] = sum(
                es.emotion_consistency(vad(a), vad(b)) for a, b in zip(s_scores, b_scores)) / len(valid)

    for k, v in metrics.items():
        print(f"  {k:40s} {v:.4f}")
    if args.json:
        existing = json.loads(args.json.read_text(encoding="utf-8")) if args.json.exists() else {}
        existing.update(metrics)
        args.json.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
    return 0


def _cuda() -> bool:
    try:
        import torch  # type: ignore

        return torch.cuda.is_available()
    except ImportError:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
