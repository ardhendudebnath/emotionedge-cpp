#!/usr/bin/env python3
"""P2 data: Hindi targets for the LoRA corpus from a larger NLLB teacher (sequence-level
distillation). The teacher reads the plain English, never the control prefix.

    python ml/data/teacher_translate.py --corpus data/mt_corpus.jsonl --out data/mt_corpus.hi.jsonl

Adds "tgt" to every line and resumes from `--out` if it already has some. Needs a GPU for a
corpus of tens of thousands of lines: NLLB-1.3B in FP16 with beam 4.

Requires: torch, transformers, sentencepiece.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

NLLB = {"en": "eng_Latn", "hi": "hin_Deva", "es": "spa_Latn", "fr": "fra_Latn", "de": "deu_Latn"}


def main() -> int:
    import torch  # type: ignore
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--teacher", default="facebook/nllb-200-distilled-1.3B")
    parser.add_argument("--src", default="en")
    parser.add_argument("--tgt", default="hi")
    parser.add_argument("--batch", type=int, default=48)
    parser.add_argument("--beam", type=int, default=4)
    args = parser.parse_args()

    items = [json.loads(line) for line in args.corpus.read_text(encoding="utf-8").splitlines() if line.strip()]
    done = {}
    if args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            done[row["id"]] = row["tgt"]
    todo = [i for i in items if i["id"] not in done]
    print(f"{len(items)} items, {len(done)} already translated, {len(todo)} to go")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.teacher, src_lang=NLLB[args.src])
    model = AutoModelForSeq2SeqLM.from_pretrained(args.teacher, dtype=torch.float16 if device == "cuda" else None)
    model = model.to(device).eval()
    forced = tokenizer.convert_tokens_to_ids(NLLB[args.tgt])

    todo.sort(key=lambda i: len(i["src"]))
    start = time.time()
    with open(args.out, "a", encoding="utf-8") as out:
        for b in range(0, len(todo), args.batch):
            batch = todo[b:b + args.batch]
            enc = tokenizer([i["src"] for i in batch], return_tensors="pt", padding=True, truncation=True,
                            max_length=200).to(device)
            with torch.inference_mode():
                gen = model.generate(**enc, forced_bos_token_id=forced, num_beams=args.beam,
                                     max_new_tokens=int(enc["input_ids"].shape[1] * 2 + 10))
            for item, text in zip(batch, tokenizer.batch_decode(gen, skip_special_tokens=True)):
                out.write(json.dumps({**item, "tgt": text}, ensure_ascii=False) + "\n")
            if (b // args.batch) % 50 == 0:
                rate = (b + len(batch)) / max(time.time() - start, 1e-6)
                print(f"  {b + len(batch)}/{len(todo)}  {rate:.0f} sentences/s")
    print(f"done in {time.time() - start:.0f} s -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
