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

Every render is played as the runtime plays it: Kokoro's ~0.4-0.5 s of silence around a clause
is trimmed, and the final contour is applied to an utterance's last clause only. Judged with the
padding, a RAVDESS sentence sounded sad (P ~ 1) under the rules and every plan, while the
pipeline heard neutral, surprise or fear; trimmed, the script hears what the pipeline hears.
With --clauses, the units are the clauses the TTS stage renders one at a time (the first cut
after 3 words), not whole sentences.

Guarding against values that only fool the training judge:
- every candidate is scored on fixed training sentences;
- the result is compared with the rules on held-out sentences by the training judge, an
  independent V·A·D model (audeering) and Whisper CER;
- an emotion keeps the rules unless, at every strength, both judges hear it better, the
  runtime's ECS rises and CER holds.

The objective (--objective) is the judge's P(emotion), its margin over the likeliest other
emotion, or the ECS the runtime would report (the judge's classes as V·A·D). The held-out report
also gives the label the runtime would put on each render. --evaluate-only compares an existing
plan with the rules; --held-texts evaluates sentences the pipeline got wrong.

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
from learn_style_offsets import EMOJI, REPO, TARGETS, VOICES, played  # noqa: E402
from eval_style_offsets import DIMENSIONAL, cer  # noqa: E402
from emotionedge_ml import emotion_space as es  # noqa: E402

CLAUSE_MARKS = (",", ";", ":", ".", "!", "?", "…", "।", "॥")  # … । ॥


def runtime_clauses(text: str, max_first_words: int = 3, min_first_chars: int = 8, min_chars: int = 24,
                    max_chars: int = 140) -> list:
    """The clauses the TTS stage renders one at a time (core/tts/clause_chunker.cpp, with
    pipeline.engines.yaml's max_first_words: 3). Lengths are in UTF-8 bytes, as there."""
    words = text.split()
    out, cur = [], []
    for i, w in enumerate(words):
        cur.append(w)
        size = len(" ".join(cur).encode("utf-8"))
        last = i + 1 == len(words)
        first_full = not out and max_first_words > 0 and len(cur) >= max_first_words and i + 2 < len(words)
        ends = w.rstrip("\"')").endswith(CLAUSE_MARKS)
        if not last and ((ends and size >= (min_first_chars if not out else min_chars)) or size >= max_chars or first_full):
            out.append(" ".join(cur))
            cur = []
    if out and cur and len(" ".join(cur).encode("utf-8")) < min_chars // 2:  # a very short tail joins the clause before
        out[-1] += " " + " ".join(cur)
        cur = []
    if cur:
        out.append(" ".join(cur))
    return out


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


class KokoroPlayback:
    """Kokoro with the runtime's prosody controls (ml/export/export_kokoro_onnx.py) and its style
    offsets, played as the runtime plays it (core/tts/kokoro_engine.cpp): the final contour on an
    utterance's last clause only, and the silence around the clause trimmed. Also used by
    eval/make_listening_test.py, so listeners hear what the search judged."""

    def __init__(self, offsets_file: Path, offset_emotions, device: str):
        import torch  # type: ignore
        from huggingface_hub import hf_hub_download  # type: ignore
        from kokoro import KModel  # type: ignore

        self.torch, self.dev = torch, device
        self.m = KModel(repo_id=REPO).to(device).eval()
        self.packs = {v: torch.load(hf_hub_download(REPO, f"voices/{v}.pt"), weights_only=True).squeeze(1).float().to(device)
                      for v in VOICES}
        offsets = json.loads(Path(offsets_file).read_text(encoding="utf-8"))["offsets"]
        self.offsets = {e: torch.tensor(offsets[e], device=device) for e in offset_emotions if e in offsets}

    def render(self, ps, voice, emotion, c, seed, strength=1.0, final=True):
        """One clause (phonemes `ps`) in `voice`, with plan `c` and `emotion`'s style offset at
        `strength`. The decoder adds noise: `seed` fixes it, for fair comparisons."""
        torch, m, dev = self.torch, self.m, self.dev
        with torch.no_grad():
            torch.manual_seed(seed)
            style = self.packs[voice][len(ps) - 1].unsqueeze(0)
            if emotion in self.offsets:
                style = style + strength * self.offsets[emotion]
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
            fall = c["fall"] if final else 0.0
            shaped = (mean * 2.0 ** (c["pitch_st"] / 12.0) + (f0 - mean) * c["range"]) * (1.0 - 0.25 * fall * ramp)
            f0 = voiced * shaped.clamp(min=10.0) + (1.0 - voiced) * f0
            audio = m.decoder(m.text_encoder(ids, lengths, mask) @ aln, f0, n, style[:, :128]).squeeze().float()
            return played(audio, final)


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
    parser.add_argument("--emotions", nargs="+", default=list(TARGETS), choices=list(TARGETS),
                        help="emotions to search; the others keep their entries in an existing --out-name")
    parser.add_argument("--clauses", action="store_true",
                        help="train and judge on the clauses the runtime renders one at a time (the first cut "
                             "after 3 words), not whole sentences")
    parser.add_argument("--objective", choices=("p", "margin", "ecs"), default="p",
                        help="p: the judge's P(emotion); margin: P(emotion) minus the likeliest other emotion; "
                             "ecs: how close what the runtime hears (the judge's classes as V·A·D) is to the emotion")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="no search: compare the plan in --out-name with the rules on held-out data")
    parser.add_argument("--start", choices=("rules", "plan"), default="rules",
                        help="where the search starts: the rules, or the plan in --out-name (to refine it)")
    parser.add_argument("--voices", nargs="+", default=list(VOICES), choices=list(VOICES),
                        help="Kokoro voices to render with (each keeps its noise seeds, so subsets compare)")
    parser.add_argument("--held-texts", nargs="+",
                        help="held-out sentences instead of the corpus slice, e.g. one the pipeline got wrong "
                             "(repeat it for more noise seeds)")
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
    def split(sentences):
        """(text, phonemes, last clause of its utterance) per unit rendered."""
        if not args.clauses:
            return [(t, p, True) for t, p in sentences]
        out = []
        for t, _ in sentences:
            parts = runtime_clauses(t)
            out += [(c, g2p(c)[0], k + 1 == len(parts)) for k, c in enumerate(parts)]
        return out

    def pick(pool, n):
        out = [(t, g2p(t)[0]) for t in pool if not EMOJI.search(t)]
        return split([(t, p) for t, p in out if 20 <= len(p) <= 200][:n])
    train = pick(texts[200:300], args.train)      # outside the offsets' training slice (first 160)
    held = (split([(t, g2p(t)[0]) for t in args.held_texts]) if args.held_texts
            else pick(texts[400:400 + args.val * 3], args.val))  # eval_style_offsets' held-out slice
    unit = "clauses" if args.clauses else "sentences"
    print(f"{len(train)} training / {len(held)} held-out {unit} x {len(args.voices)} voice(s)")

    render = KokoroPlayback(args.voice_dir / args.offsets, args.offset_emotions, dev).render

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

    others = {e: [classes.index(TARGETS[o]) for o in TARGETS if o != e] for e in TARGETS}

    def confuser(probs, emotion):
        """The likeliest other emotion's class index and P."""
        i = max(others[emotion], key=lambda c: probs[c].item())
        return i, probs[i].item()

    positions = [None if name.lower() in es.ABSTAIN_CLASSES else es.class_position(name) for name in classes]

    def heard_point(probs):
        """What the runtime's consistency stage hears in a render: the class positions weighted by
        probability over the classes that do not abstain (core/emotion/class_mapping.cpp). Its
        nearest prototype is the label; its distance to the target gives the ECS."""
        weights = [(probs[i].item(), pos) for i, pos in enumerate(positions) if pos is not None]
        evidence = sum(w for w, _ in weights)
        return [sum(w * pos[q] for w, pos in weights) / evidence for q in range(3)]

    def gain(probs, emotion):
        """One render's score under --objective. `ecs` uses the ECS's distance term itself, on
        P's scale: from 0 (heard exactly as the emotion) down to about -1 (heard as neutral or a
        neighbouring emotion)."""
        p = probs[classes.index(TARGETS[emotion])].item()
        if args.objective == "p":
            return p
        if args.objective == "margin":
            return p - confuser(probs, emotion)[1]
        return -es.distance(heard_point(probs), es.PROTOTYPES[emotion])

    def score(emotion, plan_at, sentences, reference_asr=None):
        """Judge P(emotion) averaged over the strengths, the objective's own score, and Whisper
        loss per render (for the intelligibility term). plan_at(k) is the plan the runtime
        applies at strength k."""
        target = classes.index(TARGETS[emotion])
        p, gains, losses = [], [], []
        for k in args.strengths:
            for i, (text, ps, final) in enumerate(sentences):
                for j, v in enumerate(VOICES):
                    if v not in args.voices:
                        continue
                    audio = render(ps, v, emotion, plan_at(k), seed=1000 * i + j, strength=k, final=final)
                    probs = judge_probs(audio)
                    p.append(probs[target].item())
                    gains.append(gain(probs, emotion))
                    losses.append(asr_loss(text, audio))
        p_mean, gain_mean = sum(p) / len(p), sum(gains) / len(gains)
        if reference_asr is None:
            return p_mean, losses, gain_mean
        penalty = sum(max(0.0, a - b) for a, b in zip(losses, reference_asr)) / len(losses)
        return p_mean, losses, gain_mean - args.asr_weight * penalty

    out_path = args.voice_dir / args.out_name
    previous = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else None
    learned, log = {}, {}
    start = time.time()
    for emotion in args.emotions:
        if (args.evaluate_only or args.start == "plan") and (previous is None or emotion not in previous["controls"]):
            raise SystemExit(f"no plan for {emotion} in {out_path}")
        if args.evaluate_only:  # the plan already learned
            learned[emotion] = previous["controls"][emotion]
            log[emotion] = {"rules": rules_at(emotion, 1.0)}
            continue
        p_rules, ref_asr, _ = score(emotion, lambda k: rules_at(emotion, k), train)
        base = rules_at(emotion, 1.0)
        # From the rules at full strength, or from the plan already learned.
        best = dict(base) if args.start == "rules" else clamp(previous["controls"][emotion])
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
        log[emotion] = {"rules": base, "train_p_rules": p_rules, "train_objective": best_obj, "evaluations": evaluations,
                        "objective": args.objective, "units": unit, "start": args.start}
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
        p, margins, p_other, shifts, cers, ecs = [], [], {i: 0.0 for i in others[emotion]}, [], [], []
        heard = {}
        for i, (text, ps, final) in enumerate(held):
            for j, v in enumerate(VOICES):
                if v not in args.voices:
                    continue
                seed = 50000 + 1000 * i + j
                audio = render(ps, v, emotion, c, seed, strength, final)
                probs = judge_probs(audio)
                p.append(probs[target].item())
                margins.append(p[-1] - confuser(probs, emotion)[1])
                for o in p_other:
                    p_other[o] += probs[o].item()
                point = heard_point(probs)
                ecs.append(es.emotion_consistency(point, es.PROTOTYPES[emotion]))
                label = es.nearest_label(point)
                heard[label] = heard.get(label, 0) + 1
                out, x16 = vad(audio)
                ref, _ = vad(render(ps, v, None, NEUTRAL, seed, final=final))  # same text, voice and noise, no emotion
                shifts.append([o - r for o, r in zip(out, ref)])
                cers.append(min(1.0, cer(text, asr({"raw": x16, "sampling_rate": 16000},
                                                   generate_kwargs={"language": "hindi", "task": "transcribe"})["text"])))
        shift = [sum(s[q] for s in shifts) / len(shifts) for q in range(3)]
        proto = es.PROTOTYPES[emotion]
        along = sum(s * pr for s, pr in zip(shift, proto)) / math.sqrt(sum(pr * pr for pr in proto))
        top = max(p_other, key=p_other.get)
        return {"p_judge": sum(p) / len(p), "margin": sum(margins) / len(margins), "confuser": classes[top],
                "p_confuser": p_other[top] / len(p), "ecs": sum(ecs) / len(ecs),
                "heard": {k: n / len(p) for k, n in sorted(heard.items())},
                "shift": shift, "shift_along_prototype": along, "cer": sorted(cers)[len(cers) // 2]}

    results, table, adopted = {}, {}, []
    print(f"\nheld-out ({unit}): judge P(target), its margin over the likeliest other emotion, the runtime's ECS | "
          "independent V·A·D shift vs neutral render (projection on the prototype) | median CER | runtime labels")
    for emotion in args.emotions:
        results[emotion], adopt = {}, True
        for k in args.strengths:
            r = held_out(emotion, rules_at(emotion, k), k)
            lr = held_out(emotion, scaled(learned[emotion], k), k)
            results[emotion][f"{k:g}"] = {"rules": r, "learned": lr}
            # Adopt only where, at every strength, both judges hear the emotion better on
            # held-out data (the independent one guards against fooling the training judge,
            # which is also the runtime's consistency judge), the runtime's ECS rises (so the
            # emotion is not won at the price of a neighbouring one, as sadness once was by
            # fear), and intelligibility holds.
            adopt = adopt and (lr["shift_along_prototype"] > r["shift_along_prototype"]
                               and lr["p_judge"] > r["p_judge"] and lr["ecs"] > r["ecs"]
                               and lr["cer"] <= r["cer"] + 0.02)
            for name, x in (("rules", r), ("learned", lr)):
                print(f"  {emotion:8s} x{k:<4g} {name:7s} P {x['p_judge']:.3f} margin {x['margin']:+.3f} "
                      f"(next: {x['confuser']} {x['p_confuser']:.3f}) ECS {x['ecs']:.3f} | dV {x['shift'][0]:+.3f} "
                      f"dA {x['shift'][1]:+.3f} dD {x['shift'][2]:+.3f} (along {x['shift_along_prototype']:+.3f}) "
                      f"| CER {x['cer']:.3f} | heard as " + ", ".join(f"{k} {v:.2f}" for k, v in x["heard"].items()))
        table[emotion] = learned[emotion] if adopt else log[emotion]["rules"]
        if adopt:
            adopted.append(emotion)
        print(f"  {emotion:8s} -> {'learned' if adopt else 'rules'}")
    if args.evaluate_only:
        return 0

    # The runtime applies `controls` only for the emotions in `adopted`; the rest keep the rules.
    # Emotions not searched this time keep their entries.
    out = previous if previous is not None and set(args.emotions) != set(TARGETS) else {
        "adopted": [], "controls": {}, "search": {}, "held_out": {}}
    out.update({"judge": "emotion2vec/emotion2vec_plus_base", "independent_judge": DIMENSIONAL,
                "offsets": args.offsets, "offset_emotions": args.offset_emotions, "strengths": args.strengths})
    out["adopted"] = [e for e in TARGETS if e in adopted or (e not in args.emotions and e in out["adopted"])]
    for emotion in args.emotions:
        out["controls"][emotion] = {k: round(v, 4) for k, v in table[emotion].items()}
        out["search"][emotion] = log[emotion]
        out["held_out"][emotion] = results[emotion]
    out_path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


def fmt(c: dict) -> str:
    return f"speed {c['speed']:.2f} pitch {c['pitch_st']:+.1f} st range {c['range']:.2f} fall {c['fall']:+.2f}"


if __name__ == "__main__":
    raise SystemExit(main())
