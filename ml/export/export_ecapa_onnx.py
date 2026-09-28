#!/usr/bin/env python3
"""P4 "Export": the ECAPA-TDNN speaker encoder (blueprint 1.4, 192-d voice print) from
SpeechBrain's spkrec-ecapa-voxceleb, as one ONNX graph for core/audio/ecapa_encoder.cpp:

    input  "waveform"  float32 [1, samples]  16 kHz mono
    output "embedding" float32 [192]          L2-normalized

The graph is SpeechBrain's encode_batch: an 80-band log-mel Fbank, per-utterance mean
normalization, then ECAPA-TDNN. ONNX has no complex tensors, so SpeechBrain's torch.stft is
replaced by an exact conv-based STFT (periodic Hamming window, zero-padded center). The script
checks the export against SpeechBrain on real clips.

    python ml/export/export_ecapa_onnx.py --out ~/ee-models/speaker/ecapa-voxceleb --check data/ravdess.jsonl

Requires: torch, speechbrain, onnx, onnxruntime, soundfile.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SOURCE = "speechbrain/spkrec-ecapa-voxceleb"


def main() -> int:
    import numpy as np  # type: ignore
    import onnxruntime as ort  # type: ignore
    import soundfile as sf  # type: ignore
    import torch  # type: ignore
    import torch.nn.functional as F  # type: ignore
    from speechbrain.inference.speaker import EncoderClassifier  # type: ignore

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check", type=Path, required=True, help="JSONL items with 'audio' paths")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    enc = EncoderClassifier.from_hparams(source=SOURCE, savedir=str(Path.home() / "ee-ml" / "sb-ecapa"),
                                         run_opts={"device": "cpu"})
    fbank = enc.mods.compute_features
    stft = fbank.compute_STFT
    n_fft, hop, win = stft.n_fft, stft.hop_length, stft.win_length

    class ConvStft(torch.nn.Module):
        """torch.stft(center=True, pad_mode='constant', onesided=True) as conv1d; output
        [batch, frames, bins, 2] like speechbrain.processing.features.STFT."""

        def __init__(self) -> None:
            super().__init__()
            window = stft.window.double().numpy()
            if len(window) < n_fft:  # torch centres a shorter window inside n_fft
                left = (n_fft - len(window)) // 2
                window = np.pad(window, (left, n_fft - len(window) - left))
            n, k = np.arange(n_fft), np.arange(n_fft // 2 + 1)
            angle = 2 * np.pi * np.outer(k, n) / n_fft
            self.register_buffer("re", torch.tensor(np.cos(angle) * window, dtype=torch.float32).unsqueeze(1))
            self.register_buffer("im", torch.tensor(-np.sin(angle) * window, dtype=torch.float32).unsqueeze(1))

        def forward(self, x):
            x = F.pad(x, (n_fft // 2, n_fft // 2)).unsqueeze(1)  # zero padding, like pad_mode='constant'
            re = F.conv1d(x, self.re, stride=hop).transpose(1, 2)
            im = F.conv1d(x, self.im, stride=hop).transpose(1, 2)
            return torch.stack([re, im], dim=-1)

    reference_stft = stft
    conv = ConvStft()
    x = torch.randn(1, 16000)
    err = float((conv(x) - reference_stft(x)).abs().max())
    print(f"conv STFT vs SpeechBrain STFT: max error {err:.1e} (window {win}, n_fft {n_fft}, hop {hop})")
    if err > 1e-3:
        raise SystemExit("conv STFT does not match")
    fbank.compute_STFT = conv

    class Ecapa(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features, self.norm, self.model = fbank, enc.mods.mean_var_norm, enc.mods.embedding_model

        def forward(self, waveform):
            lengths = torch.ones(1)
            feats = self.norm(self.features(waveform), lengths)
            emb = self.model(feats, lengths).reshape(-1)
            return emb / emb.norm().clamp(min=1e-12)

    model = Ecapa().eval()
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "model.onnx"
    with torch.no_grad():
        torch.onnx.export(model, (torch.randn(1, 32000) * 0.1,), str(path), input_names=["waveform"],
                          output_names=["embedding"], dynamic_axes={"waveform": {1: "samples"}},
                          opset_version=args.opset, dynamo=False)

    items = [json.loads(line) for line in args.check.read_text(encoding="utf-8").splitlines() if line.strip()][::97][:12]
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    fbank.compute_STFT = reference_stft  # SpeechBrain as shipped, for the reference embeddings
    cosines = []
    for it in items:
        audio, rate = sf.read(it["audio"], dtype="float32")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if rate != 16000:
            audio = np.interp(np.arange(0, len(audio), rate / 16000), np.arange(len(audio)), audio).astype(np.float32)
        ref = enc.encode_batch(torch.from_numpy(audio).unsqueeze(0)).reshape(-1)
        ref = (ref / ref.norm()).numpy()
        got = session.run(["embedding"], {"waveform": audio[None]})[0]
        cosines.append(float(np.dot(ref, got)))
    print(f"onnx vs SpeechBrain on {len(cosines)} clips: min cosine {min(cosines):.5f}")
    if min(cosines) < 0.999:
        raise SystemExit("exported ECAPA disagrees with SpeechBrain")
    print(f"wrote {path} ({path.stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
