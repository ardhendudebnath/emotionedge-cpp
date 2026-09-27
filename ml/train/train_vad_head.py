#!/usr/bin/env python3
"""P2 "Train & fine-tune": emotion2vec head -> V·A·D.

Trains a small regression head on frozen utterance embeddings (e.g. 768-d emotion2vec means,
extracted once with FunASR and saved as .npy next to each clip, `<audio>.emb.npy`). The loss is
1 - CCC per axis (Lin's concordance), the standard for dimensional emotion. A second output
predicts per-axis confidence, trained to track the absolute error, which the C++ fusion (2.2)
uses as its gate.

    python ml/train/train_vad_head.py --train train.jsonl --val val.jsonl --out runs/vad_head.pt

Requires: torch, numpy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(manifest: Path):
    import numpy as np  # type: ignore

    feats, targets = [], []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        emb = Path(item["audio"] + ".emb.npy")
        if emb.exists():
            feats.append(np.load(emb).astype("float32").reshape(-1))
            targets.append([item["valence"], item["arousal"], item["dominance"]])
    if not feats:
        raise SystemExit(f"no embeddings found for {manifest} (expected <audio>.emb.npy files)")
    return np.stack(feats), np.asarray(targets, dtype="float32")


def main() -> int:
    import torch  # type: ignore
    from torch import nn  # type: ignore

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    xtr, ytr = load(args.train)
    xva, yva = load(args.val)
    dim = xtr.shape[1]

    class Head(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.body = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 256), nn.GELU(), nn.Dropout(0.2))
            self.vad = nn.Linear(256, 3)
            self.conf = nn.Linear(256, 3)

        def forward(self, x):
            h = self.body(x)
            return torch.tanh(self.vad(h)), torch.sigmoid(self.conf(h))

    def ccc_loss(pred, true):
        mp, mt = pred.mean(0), true.mean(0)
        vp, vt = pred.var(0, unbiased=False), true.var(0, unbiased=False)
        cov = ((pred - mp) * (true - mt)).mean(0)
        ccc = 2 * cov / (vp + vt + (mp - mt) ** 2 + 1e-8)
        return (1 - ccc).mean(), ccc

    model = Head()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    xtr_t, ytr_t = torch.from_numpy(xtr), torch.from_numpy(ytr)
    xva_t, yva_t = torch.from_numpy(xva), torch.from_numpy(yva)
    best = float("inf")
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(xtr_t))
        for i in range(0, len(perm), 64):
            idx = perm[i:i + 64]
            vad, conf = model(xtr_t[idx])
            loss, _ = ccc_loss(vad, ytr_t[idx])
            # Confidence learns to predict how right the V·A·D estimate is.
            target_conf = 1 - (vad.detach() - ytr_t[idx]).abs().clamp(max=1)
            loss = loss + 0.2 * nn.functional.mse_loss(conf, target_conf)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vad, _ = model(xva_t)
            val_loss, ccc = ccc_loss(vad, yva_t)
        print(f"epoch {epoch:3d}  val 1-CCC {val_loss.item():.4f}  CCC v/a/d {ccc.tolist()}")
        if val_loss.item() < best:
            best = val_loss.item()
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "dim": dim}, args.out)
    print(f"best val 1-CCC {best:.4f}, saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
