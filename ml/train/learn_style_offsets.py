#!/usr/bin/env python3
"""P2/P3 "emotion out": per-emotion offsets in Kokoro-82M's 256-d style space (blueprint 4.1
"style vector").

Kokoro renders every voice in a neutral reading style, and no style encoder was released to
make an emotional one. Its prosody controls move arousal but not valence: an acoustic emotion
classifier hears angry-sounding Kokoro as "surprised" and sad-sounding Kokoro as "neutral". This
script learns, for each emotion, one offset Δ added to every voice's style vector. The loss
is the classifier's cross-entropy for that emotion on synthesized Hindi, and Δ is kept inside
a ball of radius k times the spread of the voices' own style vectors, so the voice keeps its
identity. The TTS and the classifier are both differentiable; only Δ is trained.

Guarding against a Δ that only fools the training judge:
- the offsets are validated on held-out sentences;
- ml/eval/eval_style_offsets.py scores them with an independent V·A·D model that played no
  part in training, and checks intelligibility.

    python ml/train/learn_style_offsets.py --corpus data/mt_corpus.hi.jsonl \\
        --voice-dir ~/ee-models/tts/kokoro-82m-hi --espeak-lib .../libespeak-ng.so.1 --radius 1.0

Writes <voice-dir>/style_offsets.json, which core/tts/kokoro_engine.cpp reads.

Requires: torch (CUDA), kokoro, misaki, funasr, torchaudio.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

REPO = "hexgrad/Kokoro-82M"
VOICES = ["hf_alpha", "hf_beta", "hm_omega", "hm_psi"]
# Blueprint label -> the judge's (emotion2vec+) class.
TARGETS = {"anger": "angry", "joy": "happy", "sadness": "sad", "fear": "fearful", "surprise": "surprised"}


def main() -> int:
    import torch  # type: ignore
    import torchaudio  # type: ignore

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, required=True, help="JSONL with Hindi 'tgt' sentences")
    parser.add_argument("--voice-dir", type=Path, required=True)
    parser.add_argument("--espeak-lib", type=Path)
    parser.add_argument("--radius", type=float, default=1.0, help="Δ radius in units of the voices' style spread")
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--train", type=int, default=64)
    parser.add_argument("--val", type=int, default=16)
    parser.add_argument("--seed", type=int, default=5)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    from misaki import espeak as misaki_espeak  # type: ignore
    from phonemizer.backend.espeak.wrapper import EspeakWrapper  # type: ignore

    if args.espeak_lib:
        EspeakWrapper.set_library(str(args.espeak_lib))
        EspeakWrapper.set_data_path(str(args.espeak_lib.parent / "espeak-ng-data"))
    g2p = misaki_espeak.EspeakG2P(language="hi")

    rows = [json.loads(line) for line in args.corpus.read_text(encoding="utf-8").splitlines() if line.strip()]
    texts = sorted({r["tgt"] for r in rows if r.get("tgt") and 6 <= len(r["tgt"].split()) <= 16})
    random.shuffle(texts)
    phonemes = [p for p in (g2p(t)[0] for t in texts[:(args.train + args.val) * 2]) if 20 <= len(p) <= 200]
    train_ps, val_ps = phonemes[:args.train], phonemes[args.train:args.train + args.val]
    print(f"{len(train_ps)} training / {len(val_ps)} held-out sentences")

    from huggingface_hub import hf_hub_download  # type: ignore
    from kokoro import KModel  # type: ignore

    kmodel = KModel(repo_id=REPO).to(dev).eval()
    for p in kmodel.parameters():
        p.requires_grad_(False)
    # cuDNN backpropagates through LSTMs only in training mode. The LSTMs are single-layer (no
    # inter-layer dropout), so this changes nothing else; everything else stays in eval mode.
    for module in kmodel.modules():
        if isinstance(module, torch.nn.LSTM):
            module.train()
    packs = {v: torch.load(hf_hub_download(REPO, f"voices/{v}.pt"), weights_only=True).squeeze(1).float().to(dev)
             for v in VOICES}
    spread = torch.stack([packs[v][r] for v in VOICES for r in range(20, 200, 10)]).std(0).norm().item()
    radius = args.radius * spread
    print(f"style spread across voices {spread:.3f}; Δ radius {radius:.3f}")

    def synthesize(ps: str, style):  # KModel.forward_with_tokens without its no_grad
        m = kmodel
        ids = torch.tensor([[0, *[m.vocab[c] for c in ps if c in m.vocab], 0]], device=dev)
        t = ids.shape[1]
        lengths = torch.full((1,), t, device=dev, dtype=torch.long)
        mask = torch.zeros((1, t), dtype=torch.bool, device=dev)
        d_en = m.bert_encoder(m.bert(ids, attention_mask=(~mask).int())).transpose(-1, -2)
        s = style[:, 128:]
        d = m.predictor.text_encoder(d_en, s, lengths, mask)
        x, _ = m.predictor.lstm(d)
        dur = torch.round(torch.sigmoid(m.predictor.duration_proj(x)).sum(-1)).clamp(min=1).long().squeeze(0)
        frame_token = torch.repeat_interleave(torch.arange(t, device=dev), dur)
        aln = (torch.arange(t, device=dev).unsqueeze(1) == frame_token.unsqueeze(0)).float().unsqueeze(0)
        f0, n = m.predictor.F0Ntrain(d.transpose(-1, -2) @ aln, s)
        asr = m.text_encoder(ids, lengths, mask) @ aln
        return m.decoder(asr, f0, n, style[:, :128]).squeeze()

    from funasr import AutoModel  # type: ignore

    auto = AutoModel(model="emotion2vec/emotion2vec_plus_base", hub="hf", disable_update=True, device=dev)
    judge = auto.model.eval()
    for p in judge.parameters():
        p.requires_grad_(False)
    classes = [tk.split("/")[-1] for tk in auto.kwargs["tokenizer"].token_list]

    def judge_logits(audio24):
        x = torchaudio.functional.resample(audio24, 24000, 16000).unsqueeze(0)
        x = (x - x.mean(1, keepdim=True)) / torch.sqrt(x.var(1, keepdim=True, unbiased=False) + 1e-5)
        feats = judge.forward(source=x, padding_mask=None, mask=False, features_only=True, remove_extra_tokens=True)["x"]
        return judge.proj(feats.mean(1)).squeeze(0)

    deltas = {e: torch.zeros(256, device=dev, requires_grad=True) for e in TARGETS}
    opt = torch.optim.Adam(deltas.values(), lr=args.lr * spread)

    def evaluate(sentences):
        out = {}
        with torch.no_grad():
            for e, cls in TARGETS.items():
                k = classes.index(cls)
                p = []
                for ps in sentences:
                    for v in VOICES:
                        style = (packs[v][len(ps) - 1] + deltas[e]).unsqueeze(0)
                        p.append(torch.softmax(judge_logits(synthesize(ps, style)), -1)[k].item())
                out[e] = sum(p) / len(p)
        return out

    def neutral_baseline(sentences):
        with torch.no_grad():
            probs = [torch.softmax(judge_logits(synthesize(ps, packs[v][len(ps) - 1].unsqueeze(0))), -1)
                     for ps in sentences for v in VOICES]
        mean = torch.stack(probs).mean(0)
        return {e: mean[classes.index(c)].item() for e, c in TARGETS.items()}

    print("held-out P(target) without offsets:", {k: round(v, 3) for k, v in neutral_baseline(val_ps).items()})
    start = time.time()
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for e, cls in TARGETS.items():
            k = classes.index(cls)
            for _ in range(args.batch):
                ps, v = random.choice(train_ps), random.choice(VOICES)
                style = (packs[v][len(ps) - 1] + deltas[e]).unsqueeze(0)
                loss = torch.nn.functional.cross_entropy(judge_logits(synthesize(ps, style)).unsqueeze(0),
                                                         torch.tensor([k], device=dev)) / args.batch
                loss.backward()
                total += loss.item()
        opt.step()
        with torch.no_grad():  # stay inside the ball: the voice must stay itself
            for d in deltas.values():
                norm = d.norm()
                if norm > radius:
                    d.mul_(radius / norm)
        if step % 25 == 0 or step == 1:
            print(f"  step {step}/{args.steps}  loss {total / len(TARGETS):.3f}  {(time.time() - start) / step:.1f} s/step")
    held_out = evaluate(val_ps)
    print("held-out P(target) with offsets:", {k: round(v, 3) for k, v in held_out.items()})

    out = {"judge": "emotion2vec/emotion2vec_plus_base", "radius": args.radius, "spread": spread,
           "held_out_target_probability": held_out,
           "offsets": {e: [round(float(x), 6) for x in d.detach().cpu()] for e, d in deltas.items()}}
    (args.voice_dir / "style_offsets.json").write_text(json.dumps(out) + "\n", encoding="utf-8")
    print(f"wrote {args.voice_dir / 'style_offsets.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
