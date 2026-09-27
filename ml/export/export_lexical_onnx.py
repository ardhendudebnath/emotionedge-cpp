#!/usr/bin/env python3
"""P4 "Export": the lexical emotion classifier (blueprint 2.2, DistilRoBERTa) as the model
directory that core/emotion/onnx_lexical.cpp loads:

    model.onnx      inputs input_ids, attention_mask (int64 [1, tokens]); output logits [1, classes]
    tokenizer.json  Hugging Face byte-level BPE tokenizer (the C++ side re-implements it)
    labels.json     class -> V·A·D map (emotionedge_ml.emotion_space.label_map)

    python ml/export/export_lexical_onnx.py --out ~/ee-models/emotion/distilroberta-emotion-en
    python ml/export/export_lexical_onnx.py --out DIR --golden tests/golden/roberta_tokens.json

ONNX Runtime's output is checked against PyTorch before anything is written. `--int8` also
writes a dynamically quantized model and keeps it as model.onnx only if it agrees with FP32 on
the check sentences.

Requires: torch, transformers, onnx, onnxruntime.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emotionedge_ml.emotion_space import label_map  # noqa: E402

MODEL = "j-hartmann/emotion-english-distilroberta-base"

# Spoken-style check sentences; also the tokenizer golden set (punctuation, contractions,
# casing, numbers, curly quotes, non-ASCII).
SENTENCES = [
    "I can't believe you did this!",
    "I'm so sorry, I didn't mean to hurt you.",
    "This is amazing, thank you so much!",
    "Ask not what your country can do for you.",
    "And so, my fellow Americans.",
    "Please leave the report on my desk by 5 pm.",
    "WHY would you DO that?!",
    "Oh no... what happened to the car?",
    "That smell is absolutely disgusting.",
    "I'm scared, don't go in there.",
    "We won the match 3-2 — unbelievable!",
    "It’s fine, I’ll manage on my own.",
    "Honestly? I don't care anymore.",
    "The café opens at nine.",
    "  leading and trailing spaces  ",
    "You're late again. We'll talk later, they've said it's OK.",
]


def main() -> int:
    import numpy as np  # type: ignore
    import onnxruntime as ort  # type: ignore
    import torch  # type: ignore
    from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--int8", action="store_true", help="also try a dynamically quantized model")
    parser.add_argument("--golden", type=Path, help="write tokenizer golden ids for the C++ test here")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).eval()
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    args.out.mkdir(parents=True, exist_ok=True)
    onnx_path = args.out / "model.onnx"

    class Logits(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask):
            return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

    sample = tokenizer(SENTENCES[0], return_tensors="pt")
    torch.onnx.export(Logits().eval(), (sample["input_ids"], sample["attention_mask"]), str(onnx_path),
                      input_names=["input_ids", "attention_mask"], output_names=["logits"],
                      dynamic_axes={"input_ids": {1: "tokens"}, "attention_mask": {1: "tokens"}},
                      opset_version=args.opset, dynamo=False)

    def run(path: Path):
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        outs, start = [], time.perf_counter()
        for s in SENTENCES:
            enc = tokenizer(s, return_tensors="np")
            outs.append(session.run(["logits"], {"input_ids": enc["input_ids"].astype(np.int64),
                                                 "attention_mask": enc["attention_mask"].astype(np.int64)})[0][0])
        return np.stack(outs), (time.perf_counter() - start) * 1000 / len(SENTENCES)

    with torch.no_grad():
        reference = np.stack([model(**tokenizer(s, return_tensors="pt")).logits[0].numpy() for s in SENTENCES])
    fp32, fp32_ms = run(onnx_path)
    diff = float(np.abs(fp32 - reference).max())
    print(f"onnx fp32: max |logit diff| vs torch {diff:.2e}, {fp32_ms:.1f} ms/sentence")
    if diff > 1e-3:
        raise SystemExit("ONNX export disagrees with PyTorch")

    if args.int8:
        from onnxruntime.quantization import QuantType, quantize_dynamic  # type: ignore

        int8_path = args.out / "model.int8.onnx"
        quantize_dynamic(str(onnx_path), str(int8_path), weight_type=QuantType.QInt8)
        int8, int8_ms = run(int8_path)
        agree = float((int8.argmax(1) == fp32.argmax(1)).mean())
        print(f"onnx int8: top-1 agreement {agree:.0%}, {int8_ms:.1f} ms/sentence, "
              f"{int8_path.stat().st_size / 1e6:.0f} MB vs {onnx_path.stat().st_size / 1e6:.0f} MB")
        if agree == 1.0:
            onnx_path.unlink()
            int8_path.rename(onnx_path)
            print("keeping int8 as model.onnx")
        else:
            int8_path.unlink()
            print("int8 disagrees on a check sentence: keeping fp32")

    tokenizer.backend_tokenizer.save(str(args.out / "tokenizer.json"))
    (args.out / "labels.json").write_text(json.dumps(
        label_map(labels, languages=["en"], max_tokens=128, source=args.model), indent=2) + "\n", encoding="utf-8")
    probs = np.exp(fp32) / np.exp(fp32).sum(1, keepdims=True)
    for s, p in zip(SENTENCES, probs):
        print(f"  {labels[int(p.argmax())]:>8} {p.max():.2f}  {s!r}")

    if args.golden:
        golden = [{"text": s, "ids": tokenizer(s)["input_ids"]} for s in SENTENCES]
        args.golden.write_text(json.dumps(golden, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {args.golden}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
