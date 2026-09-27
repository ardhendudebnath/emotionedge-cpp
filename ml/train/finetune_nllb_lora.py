#!/usr/bin/env python3
"""P2 "NLLB LoRA + emotion tokens": teach NLLB-200 to condition on the runtime's control prefix.

Training pairs are JSONL: {"src", "tgt", "src_lang", "tgt_lang", "emotion", "arousal", "register"}
where `tgt` is an emotionally faithful reference translation. The source is prefixed with the
exact text the C++ translate stage emits (`<emo=anger a=0.8 reg=casual> ...`, arousal quantized
to 0.1), so the fine-tuned model reads the same prefix at runtime. After conversion to
CTranslate2, run the pipeline with:

    --set translate.control_tokens=on --set translate.arousal_step=0.1 --set translate.engine=ct2

    python ml/train/finetune_nllb_lora.py --train pairs.jsonl --out runs/nllb-emo-lora

Requires: torch, transformers, peft, sentencepiece.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

NLLB = {"en": "eng_Latn", "hi": "hin_Deva", "es": "spa_Latn", "fr": "fra_Latn", "de": "deu_Latn",
        "it": "ita_Latn", "ja": "jpn_Jpan", "zh": "zho_Hans", "pt": "por_Latn", "ar": "arb_Arab"}


def control_prefix(emotion: str, arousal: float, register: str) -> str:
    """Mirror of core/translate/control_tokens.cpp format_control_prefix (arousal step 0.1)."""
    return f"<emo={emotion} a={round(arousal, 1) + 0.0:.1f} reg={register}>"  # + 0.0: no "-0.0"


def main() -> int:
    import torch  # type: ignore
    from peft import LoraConfig, get_peft_model  # type: ignore
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--base", default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                             target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
                                             task_type="SEQ_2_SEQ_LM"))
    model.print_trainable_parameters()

    pairs = [json.loads(line) for line in args.train.read_text(encoding="utf-8").splitlines() if line.strip()]
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        for p in pairs:
            tokenizer.src_lang = NLLB[p["src_lang"]]
            tokenizer.tgt_lang = NLLB[p["tgt_lang"]]
            source = control_prefix(p["emotion"], float(p["arousal"]), p.get("register", "casual")) + " " + p["src"]
            batch = tokenizer(source, text_target=p["tgt"], return_tensors="pt", truncation=True, max_length=256)
            loss = model(**batch).loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"epoch {epoch}: mean loss {total / max(1, len(pairs)):.4f}")

    merged = model.merge_and_unload()
    args.out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"saved merged model to {args.out}; convert with ml/export/convert_nllb_ct2.sh {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
