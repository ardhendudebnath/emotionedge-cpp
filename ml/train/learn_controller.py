#!/usr/bin/env python3
"""P4 "closed loop": learn the emotion controller's prosody plan for Kokoro (blueprint 4.1,
"ECS-trained controller").

The runtime controller turns a target emotion into prosody targets. For Kokoro four of them
reach the model: speaking rate, pitch shift, pitch range and the final contour. Its rules v1
were set by hand from the prosody literature. This script searches, per emotion, the four
values at full intensity that make the runtime's own judge (emotion2vec+, the consistency stage
5.2) hear that emotion in Kokoro's Hindi. They are searched on top of the shipped style offsets,
as the runtime renders them. Rendering involves rounding, so the search is derivative-free: a
pattern search started from the rules' values, inside the ranges the TTS stays intelligible in
(speed <= 1.3, eval_tts_speed.py).

The runtime rarely applies a plan in full. It scales the plan and the style offset by how far
the target V·A·D reaches toward the emotion's prototype, gated by confidence: about 0.7 on
RAVDESS. A plan searched at full strength only made the judge hear sadness at full strength. At
0.7 it heard fear about as often, and in the pipeline it heard fear nearly every time. So every
candidate is scored, and every gate is checked, at each of --strengths.

Guarding against values that only fool the training judge:
- every candidate is scored on fixed training sentences;
- the result is compared with the rules on held-out sentences by the training judge, an
  independent V·A·D model (audeering) and Whisper CER;
- an emotion keeps the rules unless, at every strength, both judges hear it better and CER holds.

    python ml/train/learn_controller.py --corpus data/mt_corpus.hi.jsonl \\
        --voice-dir ~/ee-models/tts/kokoro-82m-hi --espeak-lib <libespeak-ng.so.1>

Writes <voice-dir>/prosody_controller.json, which the controller stage reads
(core/prosody/controller_stage.cpp: plan_model_id, learned_plan, plan_emotions). Check the
adopted emotions end to end before listing them in plan_emotions: on RAVDESS, joy passed every
check here and still showed no gain in the pipeline.

Requires: torch (CUDA), kokoro, misaki, funasr, transformers, torchaudio.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from learn_style_offsets import EMOJI, REPO, TARGETS, VOICES  # noqa: E402
from eval_style_offsets import DIMENSIONAL, cer  # noqa: E402
from emotionedge_ml import emotion_space as es  # noqa: E402

CONTROLS = ("speed", "pitch_st", "range", "fall")
NEUTRAL = {"speed": 1.0, "pitch_st": 0.0, "range": 1.0, "fall": 0.0}
BOUNDS = {"speed": (0.8, 1.3), "pitch_st": (-4.0, 4.0), "range": (0.5, 2.0), "fall": (-1.0, 1.0)}
STEPS = {"speed": 0.08, "pitch_st": 1.5, "range": 0.3, "fall": 0.4}  # initial pattern-search steps


def rules(vad) -> dict:
    """core/prosody/controller.cpp rules v1 at full confidence (Hindi profile: all 1.0)."""
    v, a, d = vad
    neg_v = max(-v, 0.0)
    pitch_pct = 19.5 * a - 10.0 * neg_v * (1.0 - a) * 0.5
    return {"speed": 1.0 + 13.0 * a / 100.0, "pitch_st": 12.0 * math.log2(max(0.25, 1.0 + pitch_pct / 100.0)),
            "range": 1.0 + 38.0 * a / 100.0, "fall": max(-1.0, min(1.0, d))}


def clamp(c: dict) -> dict:
    return {k: min(max(c[k], BOUNDS[k][0]), BOUNDS[k][1]) for k in CONTROLS}


def scaled(c: dict, k: float) -> dict:
    """A learned plan at strength k, as core/prosody/controller.cpp applies it."""
    return {"speed": 1.0 + k * (c["speed"] - 1.0), "pitch_st": k * c["pitch_st"],
            "range": 1.0 + k * (c["range"] - 1.0), "fall": k * c["fall"]}


def rules_at(emotion: str, k: float) -> dict:
    """The rules for a target k of the way to the emotion's prototype."""
    return clamp(rules([k * x for x in es.PROTOTYPES[emotion]]))


def main() -> int:
    import torch  # type: ignore
    import torchaudio  # type: ignore
    from torch import nn  # type: ignore

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--voice-dir", type=Path, required=True)
    parser.add_argument("--espeak-lib", type=Path)
    parser.add_argument("--offsets", default="style_offsets.json", help="the shipped style offsets (in --voice-dir)")
    parser.add_argument("--offset-emotions", nargs="+", default=["anger", "joy", "sadness"],
                        help="emotions whose offset the runtime applies (style_emotions)")
    parser.add_argument("--train", type=int, default=8, help="training sentences (x 4 voices) per evaluation")
    parser.add_argument("--val", type=int, default=12)
    parser.add_argument("--rounds", type=int, default=4,
                        help="pattern-search rounds (each halves the steps; finer ones moved within the judge's noise)")
    parser.add_argument("--strengths", type=float, nargs="+", default=[0.7, 1.0],
                        help="strengths at which the runtime applies the plan and the offset (0.7: RAVDESS median)")
    parser.add_argument("--asr-weight", type=float, default=1.0,
                        help="penalty on Whisper-small's loss above the rules' render (intelligibility)")
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--out-name", default="prosody_controller.json")
    args = parser.parse_args()
    random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    from misaki import espeak as misaki_espeak  # type: ignore
    from phonemizer.backend.espeak.wrapper import EspeakWrapper  # type: ignore

    if args.espeak_lib:
        EspeakWrapper.set_library(str(args.espeak_lib))
        EspeakWrapper.set_data_path(str(args.espeak_lib.parent / "espeak-ng-data"))
    g2p = misaki_espeak.EspeakG2P(language="hi")

    rows = [json.loads(line) for line in args.corpus.read_text(encoding="utf-8").splitlines() if line.strip()]
    texts = sorted({r["tgt"] for r in rows if r.get("tgt") and 6 <= len(r["tgt"].split()) <= 16})
    random.Random(5).shuffle(texts)  # the same order as learn_style_offsets / eval_style_offsets
    def pick(pool, n):
        out = [(t, g2p(t)[0]) for t in pool if not EMOJI.search(t)]
        return [(t, p) for t, p in out if 20 <= len(p) <= 200][:n]
    train = pick(texts[200:300], args.train)      # outside the offsets' training slice (first 160)
    held = pick(texts[400:400 + args.val * 3], args.val)  # eval_style_offsets' held-out slice
    print(f"{len(train)} training / {len(held)} held-out sentences x {len(VOICES)} voices")

    from huggingface_hub import hf_hub_download  # type: ignore
    from kokoro import KModel  # type: ignore

    m = KModel(repo_id=REPO).to(dev).eval()
    packs = {v: torch.load(hf_hub_download(REPO, f"voices/{v}.pt"), weights_only=True).squeeze(1).float().to(dev)
             for v in VOICES}
    offsets = json.loads((args.voice_dir / args.offsets).read_text(encoding="utf-8"))["offsets"]
    offsets = {e: torch.tensor(offsets[e], device=dev) for e in args.offset_emotions if e in offsets}

    @torch.no_grad()
    def render(ps, voice, emotion, c, seed, strength=1.0):
        """Kokoro with the runtime's prosody controls (ml/export/export_kokoro_onnx.py) and its
        style offset at `strength` (core/tts/kokoro_engine.cpp)."""
        torch.manual_seed(seed)  # the decoder adds noise: the same per candidate, for fair comparisons
        style = packs[voice][len(ps) - 1].unsqueeze(0)
        if emotion in offsets:
            style = style + strength * offsets[emotion]
        ids = torch.tensor([[0, *[m.vocab[x] for x in ps if x in m.vocab], 0]], device=dev)
        t = ids.shape[1]
        lengths = torch.full((1,), t, device=dev, dtype=torch.long)
        mask = torch.zeros((1, t), dtype=torch.bool, device=dev)
        d_en = m.bert_encoder(m.bert(ids, attention_mask=(~mask).int())).transpose(-1, -2)
        s = style[:, 128:]
        d = m.predictor.text_encoder(d_en, s, lengths, mask)
        x, _ = m.predictor.lstm(d)
        dur = torch.round(torch.sigmoid(m.predictor.duration_proj(x)).sum(-1) / c["speed"]).clamp(min=1).long().squeeze(0)
        ends = torch.cumsum(dur, 0)
        frames = torch.arange(int(ends[-1]), device=dev).unsqueeze(0)
        aln = ((frames >= (ends - dur).unsqueeze(1)) & (frames < ends.unsqueeze(1))).float().unsqueeze(0)
        f0, n = m.predictor.F0Ntrain(d.transpose(-1, -2) @ aln, s)
        voiced = (f0 > 10.0).float()
        mean = (f0 * voiced).sum() / voiced.sum().clamp(min=1.0)
        pos = torch.arange(f0.shape[-1], dtype=torch.float32, device=dev) / f0.shape[-1]
        ramp = ((pos - 0.7) / 0.3).clamp(0.0, 1.0)
        shaped = (mean * 2.0 ** (c["pitch_st"] / 12.0) + (f0 - mean) * c["range"]) * (1.0 - 0.25 * c["fall"] * ramp)
        f0 = voiced * shaped.clamp(min=10.0) + (1.0 - voiced) * f0
        audio = m.decoder(m.text_encoder(ids, lengths, mask) @ aln, f0, n, style[:, :128]).squeeze()
        return audio.float()

    from funasr import AutoModel  # type: ignore

    auto = AutoModel(model="emotion2vec/emotion2vec_plus_base", hub="hf", disable_update=True, device=dev)
    judge = auto.model.eval()
    classes = [tk.split("/")[-1] for tk in auto.kwargs["tokenizer"].token_list]

    @torch.no_grad()
    def judge_probs(audio24):
        x = torchaudio.functional.resample(audio24, 24000, 16000).unsqueeze(0)
        x = (x - x.mean(1, keepdim=True)) / torch.sqrt(x.var(1, keepdim=True, unbiased=False) + 1e-5)
        feats = judge.forward(source=x, padding_mask=None, mask=False, features_only=True, remove_extra_tokens=True)["x"]
        return torch.softmax(judge.proj(feats.mean(1)).squeeze(0), -1)

    from transformers import (WhisperForConditionalGeneration, WhisperProcessor,  # type: ignore
                              Wav2Vec2Model, Wav2Vec2PreTrainedModel, Wav2Vec2Processor, pipeline)

    wproc = WhisperProcessor.from_pretrained("openai/whisper-small")
    wproc.tokenizer.set_prefix_tokens(language="hindi", task="transcribe")
    whisper = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(dev).eval()
    start_id = whisper.config.decoder_start_token_id

    @torch.no_grad()
    def asr_loss(text, audio24):  # Whisper-small's teacher-forced loss on the sentence's own text
        x = torchaudio.functional.resample(audio24, 24000, 16000).cpu().numpy()
        feats = wproc.feature_extractor(x, sampling_rate=16000, return_tensors="pt").input_features.to(dev)
        ids = wproc.tokenizer(text).input_ids
        ids = ids[1:] if ids and ids[0] == start_id else ids
        return whisper(input_features=feats, labels=torch.tensor([ids], device=dev)).loss.item()

    def score(emotion, plan_at, sentences, reference_asr=None):
        """Judge P(emotion) averaged over the strengths, and Whisper loss per render (for the
        intelligibility term). plan_at(k) is the plan the runtime applies at strength k."""
        target = classes.index(TARGETS[emotion])
        p, losses = [], []
        for k in args.strengths:
            for i, (text, ps) in enumerate(sentences):
                for j, v in enumerate(VOICES):
                    audio = render(ps, v, emotion, plan_at(k), seed=1000 * i + j, strength=k)
                    p.append(judge_probs(audio)[target].item())
                    losses.append(asr_loss(text, audio))
        p_mean = sum(p) / len(p)
        if reference_asr is None:
            return p_mean, losses, p_mean
        penalty = sum(max(0.0, a - b) for a, b in zip(losses, reference_asr)) / len(losses)
        return p_mean, losses, p_mean - args.asr_weight * penalty

    learned, log = {}, {}
    start = time.time()
    for emotion in TARGETS:
        p_rules, ref_asr, _ = score(emotion, lambda k: rules_at(emotion, k), train)
        base = rules_at(emotion, 1.0)  # the search starts from the rules at full strength
        best = dict(base)
        _, _, best_obj = score(emotion, lambda k: scaled(best, k), train, ref_asr)
        steps = dict(STEPS)
        evaluations = 2
        for rnd in range(args.rounds):
            improved = True
            while improved:  # coordinate moves until none helps, then shrink the steps
                improved = False
                for name in CONTROLS:
                    for sign in (1.0, -1.0):
                        cand = clamp({**best, name: best[name] + sign * steps[name]})
                        if cand == best:
                            continue
                        _, _, obj = score(emotion, lambda k: scaled(cand, k), train, ref_asr)
                        evaluations += 1
                        if obj > best_obj + 1e-3:
                            best, best_obj, improved = cand, obj, True
            steps = {name: s / 2 for name, s in steps.items()}
        learned[emotion] = best
        log[emotion] = {"rules": base, "train_p_rules": p_rules, "train_objective": best_obj, "evaluations": evaluations}
        print(f"{emotion:8s} rules {fmt(base)} P {p_rules:.3f} -> learned {fmt(best)} objective {best_obj:.3f} "
              f"({evaluations} evaluations, {(time.time() - start) / 60:.1f} min)")

    # Held-out comparison: the training judge, an independent dimensional judge, Whisper CER.
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

    @torch.no_grad()
    def vad(audio24):  # audeering: arousal, dominance, valence in [0, 1]
        x16 = torchaudio.functional.resample(audio24, 24000, 16000).cpu().numpy()
        a, d, v = dim_model(processor(x16, sampling_rate=16000, return_tensors="pt").input_values.to(dev))[0].tolist()
        return (2 * v - 1, 2 * a - 1, 2 * d - 1), x16

    def held_out(emotion, c, strength):
        target = classes.index(TARGETS[emotion])
        p, shifts, cers = [], [], []
        for i, (text, ps) in enumerate(held):
            for j, v in enumerate(VOICES):
                seed = 50000 + 1000 * i + j
                audio = render(ps, v, emotion, c, seed, strength)
                p.append(judge_probs(audio)[target].item())
                out, x16 = vad(audio)
                ref, _ = vad(render(ps, v, None, NEUTRAL, seed))  # same sentence, voice and noise, no emotion
                shifts.append([o - r for o, r in zip(out, ref)])
                cers.append(min(1.0, cer(text, asr({"raw": x16, "sampling_rate": 16000},
                                                   generate_kwargs={"language": "hindi", "task": "transcribe"})["text"])))
        shift = [sum(s[q] for s in shifts) / len(shifts) for q in range(3)]
        proto = es.PROTOTYPES[emotion]
        along = sum(s * pr for s, pr in zip(shift, proto)) / math.sqrt(sum(pr * pr for pr in proto))
        return {"p_judge": sum(p) / len(p), "shift": shift, "shift_along_prototype": along,
                "cer": sorted(cers)[len(cers) // 2]}

    results, table, adopted = {}, {}, []
    print("\nheld-out: judge P(target) | independent V·A·D shift vs neutral render (projection on the prototype) | median CER")
    for emotion in TARGETS:
        results[emotion], adopt = {}, True
        for k in args.strengths:
            r = held_out(emotion, rules_at(emotion, k), k)
            lr = held_out(emotion, scaled(learned[emotion], k), k)
            results[emotion][f"{k:g}"] = {"rules": r, "learned": lr}
            # Adopt only where, at every strength, both judges hear the emotion better on
            # held-out sentences (the independent one guards against fooling the training judge,
            # which is also the runtime's consistency judge), at no intelligibility cost.
            adopt = adopt and (lr["shift_along_prototype"] > r["shift_along_prototype"]
                               and lr["p_judge"] > r["p_judge"] and lr["cer"] <= r["cer"] + 0.02)
            for name, x in (("rules", r), ("learned", lr)):
                print(f"  {emotion:8s} x{k:<4g} {name:7s} P {x['p_judge']:.3f} | dV {x['shift'][0]:+.3f} "
                      f"dA {x['shift'][1]:+.3f} dD {x['shift'][2]:+.3f} (along {x['shift_along_prototype']:+.3f}) "
                      f"| CER {x['cer']:.3f}")
        table[emotion] = learned[emotion] if adopt else log[emotion]["rules"]
        if adopt:
            adopted.append(emotion)
        print(f"  {emotion:8s} -> {'learned' if adopt else 'rules'}")

    # The runtime applies `controls` only for the emotions in `adopted`; the rest keep the rules.
    out = {"judge": "emotion2vec/emotion2vec_plus_base", "independent_judge": DIMENSIONAL,
           "offsets": args.offsets, "offset_emotions": args.offset_emotions, "strengths": args.strengths,
           "adopted": adopted,
           "controls": {e: {k: round(v, 4) for k, v in c.items()} for e, c in table.items()},
           "search": log, "held_out": results}
    (args.voice_dir / args.out_name).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.voice_dir / args.out_name}")
    return 0


def fmt(c: dict) -> str:
    return f"speed {c['speed']:.2f} pitch {c['pitch_st']:+.1f} st range {c['range']:.2f} fall {c['fall']:+.2f}"


if __name__ == "__main__":
    raise SystemExit(main())
