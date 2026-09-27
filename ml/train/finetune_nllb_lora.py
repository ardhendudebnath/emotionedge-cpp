#!/usr/bin/env python3
"""P2 "NLLB LoRA + emotion tokens": teach NLLB-200 to read the runtime's control prefix.

Training data is the JSONL from ml/data/build_mt_corpus.py plus teacher targets from
ml/data/teacher_translate.py: {"src", "tgt", "prefix", "split", ...}. The source is the exact
prefix the C++ translate stage emits (`<emo=anger a=0.8 reg=casual> ...`, arousal quantized to
0.1) followed by the text, so the fine-tuned model reads what it will see at runtime.

    python ml/train/finetune_nllb_lora.py --data data/mt_corpus.hi.jsonl --out runs/nllb-emo-lora

The best adapter (lowest validation loss) is merged into the base weights and saved to
`<out>/merged`. Convert that with ml/export/convert_nllb_ct2.sh, then run the pipeline with:

    --set translate.model_id=<its id> --set translate.control_tokens=on --set translate.arousal_step=0.1

Requires: torch (CUDA recommended), transformers, peft, sentencepiece.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

NLLB = {"en": "eng_Latn", "hi": "hin_Deva", "es": "spa_Latn", "fr": "fra_Latn", "de": "deu_Latn",
        "it": "ita_Latn", "ja": "jpn_Jpan", "zh": "zho_Hans", "pt": "por_Latn", "ar": "arb_Arab"}


def control_prefix(emotion: str, arousal: float, register: str) -> str:
    """Mirror of core/translate/control_tokens.cpp format_control_prefix (arousal step 0.1)."""
    return f"<emo={emotion} a={round(arousal, 1) + 0.0:.1f} reg={register}>"  # + 0.0: no "-0.0"


def main() -> int:
    import torch  # type: ignore
    from peft import LoraConfig, get_peft_model  # type: ignore
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup  # type: ignore

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--base", default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--src", default="en")
    parser.add_argument("--tgt", default="hi")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=32, help="max sentences per batch")
    parser.add_argument("--max-tokens", type=int, default=1536, help="max target tokens per batch (padded)")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [r for r in rows if r.get("tgt")]
    train = [r for r in rows if r.get("split") != "valid"]
    valid = [r for r in rows if r.get("split") == "valid"]
    print(f"{len(train)} train / {len(valid)} valid pairs")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.base, src_lang=NLLB[args.src], tgt_lang=NLLB[args.tgt])
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base)
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                                             target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
                                             task_type="SEQ_2_SEQ_LM"))
    model.print_trainable_parameters()
    model.to(device)

    def target_tokens(data):
        return [len(ids) for ids in tokenizer(text_target=[r["tgt"] for r in data], truncation=True,
                                              max_length=192)["input_ids"]]

    def chunks_for(data):
        # Length-bucketed batches under a target-token budget: the logits over NLLB's 256k
        # vocabulary take tokens x 256k floats, so one batch of long sentences can exhaust the GPU.
        lengths = target_tokens(data)
        order = sorted(range(len(data)), key=lambda i: lengths[i])
        chunks, current = [], []
        for i in order:
            if current and (len(current) >= args.batch or (len(current) + 1) * lengths[i] > args.max_tokens):
                chunks.append(current)
                current = []
            current.append(i)
        return chunks + ([current] if current else [])

    chunk_cache = {}

    def batches(data, shuffle):
        if id(data) not in chunk_cache:
            chunk_cache[id(data)] = chunks_for(data)
        chunks = list(chunk_cache[id(data)])
        if shuffle:
            random.shuffle(chunks)
        for chunk in chunks:
            sources = [data[i]["prefix"] + " " + data[i]["src"] for i in chunk]
            enc = tokenizer(sources, text_target=[data[i]["tgt"] for i in chunk], return_tensors="pt",
                            padding=True, truncation=True, max_length=192)
            enc["labels"][enc["labels"] == tokenizer.pad_token_id] = -100
            yield {k: v.to(device) for k, v in enc.items()}

    def validation_loss():
        model.eval()
        total, n = 0.0, 0
        with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            for batch in batches(valid, shuffle=False):
                total += model(**batch).loss.item() * len(batch["input_ids"])
                n += len(batch["input_ids"])
        model.train()
        return total / max(1, n)

    chunk_cache[id(train)] = chunks_for(train)
    steps = args.epochs * len(chunk_cache[id(train)])
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    best = validation_loss()
    print(f"validation loss before training: {best:.4f}")
    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out / "adapter")
    model.train()
    step, start = 0, time.time()
    for epoch in range(args.epochs):
        for batch in batches(train, shuffle=True):
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 100 == 0:
                print(f"  step {step}/{steps}  loss {loss.item():.4f}  {step / (time.time() - start):.1f} steps/s")
        val = validation_loss()
        print(f"epoch {epoch}: validation loss {val:.4f}")
        if val < best:
            best = val
            model.save_pretrained(args.out / "adapter")

    from peft import PeftModel  # type: ignore

    base = AutoModelForSeq2SeqLM.from_pretrained(args.base)
    merged = PeftModel.from_pretrained(base, args.out / "adapter").merge_and_unload()
    merged.save_pretrained(args.out / "merged")
    tokenizer.save_pretrained(args.out / "merged")
    print(f"best validation loss {best:.4f}; merged model in {args.out / 'merged'} "
          f"({time.time() - start:.0f} s of training)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
