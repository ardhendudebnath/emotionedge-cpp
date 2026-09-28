#!/usr/bin/env python3
"""P5 check of Kokoro's learned style offsets (ml/train/learn_style_offsets.py) with judges that
took no part in training, on sentences it never saw.

For each emotion's offset, every held-out Hindi sentence is rendered in every voice twice: with
the offset and without (the neutral render). Reported per emotion:
- the V·A·D shift measured by audeering's wav2vec2 dimensional model (MSP-Podcast), rescaled
  to [-1, 1], and whether each axis moves the way the emotion's prototype points
  (config/emotion_space.json);
- the training judge's probability for the emotion (emotion2vec+), for reference only;
- intelligibility: Whisper-small's character error rate against the Hindi text, with and without
  the offset.

    python ml/eval/eval_style_offsets.py --corpus data/mt_corpus.hi.jsonl \\
        --voice-dir ~/ee-models/tts/kokoro-82m-hi --espeak-lib .../libespeak-ng.so.1 --json out.json

Requires: torch, kokoro, misaki, funasr, transformers (audeering model, Whisper).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emotionedge_ml import emotion_space as es  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
from learn_style_offsets import EMOJI, REPO, TARGETS, VOICES  # noqa: E402

DIMENSIONAL = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"


def cer(ref: str, hyp: str) -> float:
    ref, hyp = ref.replace(" ", ""), hyp.replace(" ", "")
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1] / max(1, len(ref))


def main() -> int:
    import torch  # type: ignore
    import torchaudio  # type: ignore
    from torch import nn  # type: ignore

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--voice-dir", type=Path, required=True)
    parser.add_argument("--espeak-lib", type=Path)
    parser.add_argument("--sentences", type=int, default=12)
    parser.add_argument("--strength", type=float, default=1.0, help="scale on the offsets")
    parser.add_argument("--offsets", default="style_offsets.json", help="file in --voice-dir")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    from misaki import espeak as misaki_espeak  # type: ignore
    from phonemizer.backend.espeak.wrapper import EspeakWrapper  # type: ignore

    if args.espeak_lib:
        EspeakWrapper.set_library(str(args.espeak_lib))
        EspeakWrapper.set_data_path(str(args.espeak_lib.parent / "espeak-ng-data"))
    g2p = misaki_espeak.EspeakG2P(language="hi")

    # Held out: the learner draws from the first 160 texts of this shuffle (seed 5); take later ones.
    rows = [json.loads(line) for line in args.corpus.read_text(encoding="utf-8").splitlines() if line.strip()]
    texts = sorted({r["tgt"] for r in rows if r.get("tgt") and 6 <= len(r["tgt"].split()) <= 16})
    random.Random(5).shuffle(texts)
    held = [(t, g2p(t)[0]) for t in texts[400:400 + args.sentences * 3] if not EMOJI.search(t)]
    held = [(t, p) for t, p in held if 20 <= len(p) <= 200][:args.sentences]

    offsets = json.loads((args.voice_dir / args.offsets).read_text(encoding="utf-8"))["offsets"]
    from huggingface_hub import hf_hub_download  # type: ignore
    from kokoro import KModel  # type: ignore

    kmodel = KModel(repo_id=REPO).to(dev).eval()
    packs = {v: torch.load(hf_hub_download(REPO, f"voices/{v}.pt"), weights_only=True).squeeze(1).float().to(dev)
             for v in VOICES}

    from transformers import Wav2Vec2Model, Wav2Vec2PreTrainedModel, Wav2Vec2Processor, pipeline  # type: ignore

    class RegressionHead(nn.Module):  # audeering's model card
        def __init__(self, config):
            super().__init__()
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.dropout = nn.Dropout(config.final_dropout)
            self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        def forward(self, x):
            return self.out_proj(self.dropout(torch.tanh(self.dense(self.dropout(x)))))

    class EmotionModel(Wav2Vec2PreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.wav2vec2 = Wav2Vec2Model(config)
            self.classifier = RegressionHead(config)
            self.init_weights()

        def forward(self, input_values):
            return self.classifier(self.wav2vec2(input_values)[0].mean(dim=1))

    processor = Wav2Vec2Processor.from_pretrained(DIMENSIONAL)
    dim_model = EmotionModel.from_pretrained(DIMENSIONAL).to(dev).eval()
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-small", device=0 if dev == "cuda" else -1)

    from funasr import AutoModel  # type: ignore

    auto = AutoModel(model="emotion2vec/emotion2vec_plus_base", hub="hf", disable_update=True, device=dev)
    classes = [tk.split("/")[-1] for tk in auto.kwargs["tokenizer"].token_list]

    def render(ps, voice, delta):
        style = packs[voice][len(ps) - 1].unsqueeze(0)
        if delta is not None:
            style = style + args.strength * torch.tensor(delta, device=dev)
        with torch.no_grad():
            audio, _ = kmodel.forward_with_tokens(
                torch.tensor([[0, *[kmodel.vocab[c] for c in ps if c in kmodel.vocab], 0]], device=dev), style, 1.0)
        return torchaudio.functional.resample(audio.float().cpu(), 24000, 16000).numpy()

    def vad(audio16):  # audeering outputs arousal, dominance, valence in [0, 1]
        x = processor(audio16, sampling_rate=16000, return_tensors="pt").input_values.to(dev)
        with torch.no_grad():
            a, d, v = dim_model(x)[0].tolist()
        return (2 * v - 1, 2 * a - 1, 2 * d - 1)

    def probability(audio16, cls):
        res = auto.generate(audio16, granularity="utterance", extract_embedding=False, disable_pbar=True)[0]
        return dict(zip((lb.split("/")[-1] for lb in res["labels"]), res["scores"])).get(cls, 0.0)

    def transcribe(audio16):
        return asr({"raw": audio16, "sampling_rate": 16000}, generate_kwargs={"language": "hindi", "task": "transcribe"})["text"]

    neutral = {}
    for text, ps in held:
        for v in VOICES:
            a = render(ps, v, None)
            neutral[(ps, v)] = (vad(a), cer(text, transcribe(a)))
    base_cer = sum(c for _, c in neutral.values()) / len(neutral)
    metrics = {"cer_neutral": base_cer}
    print(f"{len(held)} held-out sentences x {len(VOICES)} voices; neutral render CER {base_cer:.3f}")
    for emotion, delta in offsets.items():
        shifts, cers, probs = [], [], []
        for text, ps in held:
            for v in VOICES:
                a = render(ps, v, delta)
                out = vad(a)
                shifts.append([o - n for o, n in zip(out, neutral[(ps, v)][0])])
                cers.append(cer(text, transcribe(a)))
                probs.append(probability(a, TARGETS[emotion]))
        mean = [sum(s[k] for s in shifts) / len(shifts) for k in range(3)]
        proto = es.PROTOTYPES[emotion]
        axes = [k for k in range(3) if abs(proto[k]) >= 0.2]
        agree = sum((mean[k] > 0) == (proto[k] > 0) for k in axes) / len(axes)
        metrics.update({f"shift_{emotion}_v": mean[0], f"shift_{emotion}_a": mean[1], f"shift_{emotion}_d": mean[2],
                        f"direction_agreement_{emotion}": agree, f"cer_{emotion}": sum(cers) / len(cers),
                        f"train_judge_p_{emotion}": sum(probs) / len(probs)})
        print(f"  {emotion:8s} dV {mean[0]:+.3f} dA {mean[1]:+.3f} dD {mean[2]:+.3f} "
              f"(prototype {proto[0]:+.2f} {proto[1]:+.2f} {proto[2]:+.2f}; direction agreement {agree:.0%}) | "
              f"emotion2vec P({TARGETS[emotion]}) {sum(probs) / len(probs):.2f} | CER {sum(cers) / len(cers):.3f}")
    if args.json:
        args.json.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
