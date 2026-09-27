#!/usr/bin/env python3
"""P4 "Export": the acoustic emotion model as one ONNX graph with the runtime's contract
(core/emotion/onnx_acoustic.cpp):

    input  "waveform"   float32 [1, samples]  16 kHz mono in [-1, 1]
    output "vad"        float32 [1, 3]        valence, arousal, dominance in [-1, 1]
    output "confidence" float32 [1, 3]        per-axis confidence in [0, 1]

The encoder is any Hugging Face audio encoder that yields frame embeddings (a wav2vec2/data2vec
style model, e.g. an emotion2vec port); its mean-pooled output feeds the head trained by
ml/train/train_vad_head.py. Then register it:

    python ml/export/export_acoustic_onnx.py --encoder <hf-model> --head runs/vad_head.pt \\
        --out models/emotion/acoustic.onnx
    python ml/export/write_manifest.py add --id emotion.acoustic.v1 --task emotion.acoustic \\
        --format onnx --path emotion/acoustic.onnx

and run the pipeline with `--set emotion.acoustic=onnx --set emotion.acoustic_model_id=emotion.acoustic.v1`.

Requires: torch, transformers, onnx.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    import torch  # type: ignore
    from torch import nn  # type: ignore
    from transformers import AutoModel  # type: ignore

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--encoder", required=True, help="Hugging Face audio encoder id or path")
    parser.add_argument("--head", type=Path, required=True, help="checkpoint from train_vad_head.py")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    encoder = AutoModel.from_pretrained(args.encoder)
    checkpoint = torch.load(args.head, map_location="cpu")
    dim = checkpoint["dim"]

    class Head(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.body = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 256), nn.GELU(), nn.Dropout(0.2))
            self.vad = nn.Linear(256, 3)
            self.conf = nn.Linear(256, 3)

        def forward(self, x):
            h = self.body(x)
            return torch.tanh(self.vad(h)), torch.sigmoid(self.conf(h))

    head = Head()
    head.load_state_dict(checkpoint["state_dict"])

    class AcousticEmotion(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = encoder
            self.head = head

        def forward(self, waveform):
            frames = self.encoder(waveform).last_hidden_state  # [1, T, dim]
            return self.head(frames.mean(dim=1))

    model = AcousticEmotion().eval()
    dummy = torch.zeros(1, 16000)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, (dummy,), str(args.out), input_names=["waveform"], output_names=["vad", "confidence"],
                      dynamic_axes={"waveform": {1: "samples"}}, opset_version=args.opset)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
