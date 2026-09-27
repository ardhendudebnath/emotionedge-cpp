#!/usr/bin/env python3
"""P4 "Export": emotion2vec+ (blueprint 2.2 "ACOUSTIC: emotion2vec") as the classifier directory
that core/emotion/onnx_acoustic.cpp loads:

    model.onnx   input "waveform" float32 [1, samples] (16 kHz mono, [-1, 1]); output "logits" [1, classes]
    labels.json  class -> V·A·D map (emotionedge_ml.emotion_space.label_map)

The graph is FunASR's own utterance-level inference: the per-utterance waveform normalization,
data2vec features, mean pooling and the classifier. Classes FunASR masks ("unuse_*") get -inf
logits. The export is checked against FunASR's scores on real clips.

    python ml/export/export_emotion2vec_onnx.py --out ~/ee-models/emotion/emotion2vec-plus-base \\
        --check data/ravdess.jsonl [--int8]

`--int8` quantizes the transformer MatMuls dynamically and keeps the result as model.onnx only if
its top-1 agrees with FP32 on at least 97% of the check clips.

Requires: torch, funasr, onnx, onnxruntime, soundfile.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emotionedge_ml.emotion_space import label_map  # noqa: E402

MODEL = "emotion2vec/emotion2vec_plus_base"


def main() -> int:
    import numpy as np  # type: ignore
    import onnxruntime as ort  # type: ignore
    import soundfile as sf  # type: ignore
    import torch  # type: ignore
    from funasr import AutoModel  # type: ignore

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check", type=Path, required=True, help="JSONL items with 16 kHz-readable 'audio'")
    parser.add_argument("--clips", type=int, default=64, help="check clips (spread over the file)")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--int8", action="store_true")
    args = parser.parse_args()

    auto = AutoModel(model=args.model, hub="hf", disable_update=True, device="cpu")
    net = auto.model.eval()
    tokens = list(auto.kwargs["tokenizer"].token_list)
    names = [t.split("/")[-1] for t in tokens]  # "生气/angry" -> "angry"
    masked = torch.tensor([t.startswith("unuse") for t in tokens])
    normalize = bool(net.cfg.get("normalize", False)) if hasattr(net.cfg, "get") else bool(net.cfg.normalize)
    print(f"classes {names}; normalize={normalize}")

    class Emotion2vecLogits(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = net

        def forward(self, waveform):
            x = waveform
            if normalize:  # F.layer_norm over the whole utterance, as FunASR's inference does
                mean = x.mean(dim=1, keepdim=True)
                var = x.var(dim=1, keepdim=True, unbiased=False)
                x = (x - mean) / torch.sqrt(var + 1e-5)
            feats = self.net.forward(source=x, padding_mask=None, mask=False, features_only=True,
                                     remove_extra_tokens=True)["x"]
            logits = self.net.proj(feats.mean(dim=1))
            return logits.masked_fill(masked.unsqueeze(0), float("-inf"))

    args.out.mkdir(parents=True, exist_ok=True)
    onnx_path = args.out / "model.onnx"
    with torch.no_grad():
        torch.onnx.export(Emotion2vecLogits().eval(), (torch.randn(1, 32000) * 0.1,), str(onnx_path),
                          input_names=["waveform"], output_names=["logits"],
                          dynamic_axes={"waveform": {1: "samples"}}, opset_version=args.opset, dynamo=False)

    import torchaudio  # type: ignore

    def load_16k(path: str):
        audio, rate = sf.read(path, dtype="float32")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if rate != 16000:  # RAVDESS is 48 kHz
            audio = torchaudio.functional.resample(torch.from_numpy(audio), rate, 16000).numpy()
        return np.ascontiguousarray(audio, dtype=np.float32)

    items = [json.loads(line) for line in args.check.read_text(encoding="utf-8").splitlines() if line.strip()]
    step = max(1, len(items) // args.clips)
    clips = [load_16k(it["audio"]) for it in items[::step][:args.clips]]

    def funasr_probs(wav):
        res = auto.generate(wav, granularity="utterance", extract_embedding=False, disable_pbar=True)[0]
        by_name = {lb.split("/")[-1]: s for lb, s in zip(res["labels"], res["scores"])}
        return np.array([by_name.get(n, 0.0) for n in names])

    def onnx_run(path: Path):
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        probs, ms = [], []
        for c in clips:
            start = time.perf_counter()
            logits = session.run(["logits"], {"waveform": c[None, :].astype(np.float32)})[0][0]
            ms.append((time.perf_counter() - start) * 1000)
            e = np.exp(logits - logits[np.isfinite(logits)].max())
            probs.append(np.where(np.isfinite(logits), e, 0.0) / np.where(np.isfinite(logits), e, 0.0).sum())
        return np.stack(probs), ms

    reference = np.stack([funasr_probs(c) for c in clips])
    fp32, fp32_ms = onnx_run(onnx_path)
    diff = float(np.abs(fp32 - reference).max())
    agree = float((fp32.argmax(1) == reference.argmax(1)).mean())
    seconds = np.array([len(c) / 16000 for c in clips])
    print(f"onnx fp32 vs funasr: max |prob diff| {diff:.2e}, top-1 agreement {agree:.0%}; "
          f"{np.mean(fp32_ms):.0f} ms per clip (mean clip {seconds.mean():.1f} s)")
    if diff > 1e-3:
        raise SystemExit("ONNX export disagrees with FunASR")

    if args.int8:
        from onnxruntime.quantization import QuantType, quantize_dynamic  # type: ignore

        int8_path = args.out / "model.int8.onnx"
        quantize_dynamic(str(onnx_path), str(int8_path), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"])
        int8, int8_ms = onnx_run(int8_path)
        agree8 = float((int8.argmax(1) == fp32.argmax(1)).mean())
        print(f"onnx int8: top-1 agreement with fp32 {agree8:.0%}, max |prob diff| {np.abs(int8 - fp32).max():.2f}, "
              f"{np.mean(int8_ms):.0f} ms per clip, {int8_path.stat().st_size / 1e6:.0f} MB "
              f"vs {onnx_path.stat().st_size / 1e6:.0f} MB")
        if agree8 >= 0.97:
            onnx_path.unlink()
            int8_path.rename(onnx_path)
            print("keeping int8 as model.onnx")
        else:
            int8_path.unlink()
            print("int8 changes too many decisions: keeping fp32")

    (args.out / "labels.json").write_text(json.dumps(
        label_map(names, languages=["*"], source=args.model), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
