#!/usr/bin/env python3
"""P4: compares Hindi training targets for the MT LoRA, e.g. the P2 teacher's ("tgt") against a local
LLM's emotion-faithful ones ("tgt_llm", ml/data/llm_translate.py), on the items that have all of them.
Each target is back-translated to English with --backtranslate (vanilla NLLB), then:
- meaning: chrF of the round trip against the English source;
- emotion: DistilRoBERTa's label on the round trip, against its label on the source (as eval_mt.py's
  round trip) and against the item's own emotion label, plus the mean text ECS (1 - |dVAD| / 2 sqrt 3).

    python ml/eval/eval_mt_targets.py --corpus data/mt_corpus.llm.jsonl --fields tgt tgt_llm \\
        --backtranslate models/mt/nllb-200-distilled-600M-int8 [--json targets.json]

Requires: ctranslate2, sentencepiece, sacrebleu, transformers, torch.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_mt import CLASSIFIER, Ct2Nllb, _cuda, es  # noqa: E402


def main() -> int:
    import sacrebleu  # type: ignore
    from transformers import pipeline  # type: ignore

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--fields", nargs="+", default=["tgt", "tgt_llm"])
    parser.add_argument("--backtranslate", type=Path, required=True, help="CT2 NLLB model for hin->eng round trips")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.corpus.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [r for r in rows if all(r.get(f) for f in args.fields)]
    print(f"{len(rows)} items with every field ({', '.join(args.fields)})")
    back = Ct2Nllb(args.backtranslate)
    classify = pipeline("text-classification", model=CLASSIFIER, top_k=None, device=0 if _cuda() else -1)
    labels = [s["label"] for s in classify(["hello"])[0]]
    positions = {lb: es.class_position(lb) for lb in labels}

    def vad(scores):
        return tuple(sum(s["score"] * positions[s["label"]][k] for s in scores) for k in range(3))

    def top(scores):
        return es.DATASET_LABELS[max(scores, key=lambda s: s["score"])["label"]]

    sources = [r["src"] for r in rows]
    s_scores = classify(sources, batch_size=64, truncation=True)
    metrics: dict = {"items": len(rows)}
    for field in args.fields:
        trip = back.translate([r[field] for r in rows], [""] * len(rows), src="hin_Deva", tgt="eng_Latn")
        b_scores = classify(trip, batch_size=64, truncation=True)
        by_label: dict[str, list[bool]] = defaultdict(list)
        for r, b in zip(rows, b_scores):
            by_label[r["emotion"]].append(top(b) == r["emotion"])
        m = {"roundtrip_chrf": sacrebleu.corpus_chrf(trip, [sources]).score,
             "agreement_with_source": sum(top(a) == top(b) for a, b in zip(s_scores, b_scores)) / len(rows),
             "agreement_with_label": sum(top(b) == r["emotion"] for r, b in zip(rows, b_scores)) / len(rows),
             "text_ecs": sum(es.emotion_consistency(vad(a), vad(b)) for a, b in zip(s_scores, b_scores)) / len(rows),
             "agreement_with_label_by_emotion": {k: round(sum(v) / len(v), 3) for k, v in sorted(by_label.items())},
             "examples": [{"src": r["src"], "emotion": r["emotion"], "hi": r[field], "back": t}
                          for r, t in list(zip(rows, trip))[:8]]}
        metrics[field] = m
        print(f"{field:10s} round-trip chrF {m['roundtrip_chrf']:.1f}  emotion agreement: with source "
              f"{m['agreement_with_source']:.3f}, with label {m['agreement_with_label']:.3f}  text ECS {m['text_ecs']:.3f}")
        print(f"           by label: {m['agreement_with_label_by_emotion']}")
    if args.json:
        args.json.write_text(json.dumps(metrics, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
