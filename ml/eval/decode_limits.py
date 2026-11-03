#!/usr/bin/env python3
"""How long NLLB's translations into Hindi run from every source language the runtime maps
(core/translate/languages.cpp), and what the runtime's decode limits (core/translate/
decode_limits.hpp) do to them. NLLB is fed as core/translate/ct2_translator.cpp feeds it: the
neutral control prefix as its own pieces (--prefix runtime), beam 2, no length cap below --cap.

    python ml/eval/decode_limits.py --model ~/ee-models/mt/nllb-200-distilled-600M-emo-int8 \\
        --flores data/flores200_dataset --json decode_limits.json

- finals: FLORES-200 devtest sentences (all, or --sample spread out plus the 30 longest);
- drafts: the first half of each, in words (characters in the scripts the runtime splits by
  character), as a caption draft translates a partial transcript's stable start;
- loops: outputs that did not end within --cap, or whose distinct 4-grams of pieces are under
  half of all their 4-grams (a repetition loop; real translations score near 1);
- each candidate limit, a x source pieces + b: the loops and the other decodes it would cut, and
  the least room it leaves the others. A hypothesis is the language token, the output and the
  end token; the limit is 1 + ceil(a x src + b), as DecodeLimits counts it;
- every final the runtime's limit (--final) cuts, decoded again at that limit with the
  runtime's retry penalty (--retry-penalty): whether the retry ends.

Requires: ctranslate2, sentencepiece.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# core/translate/languages.cpp, Hindi (the target) aside.
CODES = ["eng_Latn", "spa_Latn", "fra_Latn", "deu_Latn", "ita_Latn", "por_Latn", "rus_Cyrl", "jpn_Jpan", "zho_Hans",
         "kor_Hang", "arb_Arab", "ben_Beng", "tam_Taml", "tel_Telu", "mar_Deva", "urd_Arab", "guj_Gujr", "pan_Guru",
         "kan_Knda", "mal_Mlym", "tur_Latn", "vie_Latn", "ind_Latn", "nld_Latn", "pol_Latn", "ukr_Cyrl", "swe_Latn",
         "tha_Thai", "heb_Hebr", "pes_Arab", "swh_Latn", "ell_Grek", "ces_Latn", "ron_Latn", "hun_Latn", "fin_Latn",
         "dan_Latn", "nob_Latn", "zsm_Latn", "npi_Deva", "sin_Sinh", "mya_Mymr", "khm_Khmr", "lao_Laoo"]
# languages.cpp kNoSpaces: split into words by character.
BY_CHARACTER = {"zho_Hans", "jpn_Jpan", "tha_Thai", "mya_Mymr", "khm_Khmr", "lao_Laoo"}
NEUTRAL_PREFIX = "<emo=neutral a=0.0 reg=casual>"


def distinct4(pieces: list[str]) -> float:
    grams = [tuple(pieces[i:i + 4]) for i in range(len(pieces) - 3)]
    return len(set(grams)) / len(grams) if grams else 1.0


def first_half(text: str, code: str) -> str:
    if code in BY_CHARACTER:
        return text[:max(1, len(text) // 2)]
    words = text.split()
    return " ".join(words[:max(1, len(words) // 2)])


def runtime_limit(a: float, b: float, src: int) -> int:
    """Target tokens after the language token, the end token included."""
    return math.ceil(a * src + b)


class Nllb:
    def __init__(self, model_dir: Path, device: str, beam: int, prefix: str) -> None:
        import ctranslate2  # type: ignore
        import sentencepiece as spm  # type: ignore

        if device == "auto":
            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        compute_type = "int8_float16" if device == "cuda" else "int8"
        self.translator = ctranslate2.Translator(str(model_dir), device=device, compute_type=compute_type)
        self.sp = spm.SentencePieceProcessor(model_file=str(model_dir / "sentencepiece.bpe.model"))
        self.beam = beam
        self.control = self.sp.encode(NEUTRAL_PREFIX, out_type=str) if prefix == "runtime" else []
        self.setting = f"{device} {compute_type}, beam {beam}, prefix {prefix}"

    def pieces(self, text: str) -> list[str]:
        return self.sp.encode(text, out_type=str)

    def decode(self, code: str, texts: list[str], cap: int, penalty: float = 1.0):
        """(output pieces, ended) per text, the language and end tokens left out."""
        tokens = [[code] + self.control + self.pieces(t) + ["</s>"] for t in texts]
        results = self.translator.translate_batch(tokens, target_prefix=[["hin_Deva"]] * len(tokens),
                                                  beam_size=self.beam, max_decoding_length=cap, return_end_token=True,
                                                  repetition_penalty=penalty, max_batch_size=32)
        out = []
        for r in results:
            hyp = r.hypotheses[0][1:]
            ended = bool(hyp) and hyp[-1] == "</s>"
            out.append((hyp[:-1] if ended else hyp, ended))
        return out


def pick(lines: list[str], sample: int) -> list[int]:
    if sample <= 0:
        return list(range(len(lines)))
    step = max(1, len(lines) // sample)
    longest = sorted(range(len(lines)), key=lambda i: -len(lines[i]))[:30]
    return sorted(set(range(0, len(lines), step)[:sample]) | set(longest))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True, help="CTranslate2 NLLB export with sentencepiece.bpe.model")
    ap.add_argument("--flores", type=Path, required=True, help="flores200_dataset (devtest/<code>.devtest)")
    ap.add_argument("--json", type=Path, help="write every decode's lengths, the loops and the summary here")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--beam", type=int, default=2)
    ap.add_argument("--prefix", choices=["runtime", "none"], default="runtime",
                    help="runtime: the neutral control prefix (the emotion LoRA); none: plain text (vanilla NLLB)")
    ap.add_argument("--cap", type=int, default=1024, help="decode cap: a decode still going here never ended")
    ap.add_argument("--sample", type=int, default=0, help="per language: this many spread out + the 30 longest (0: all)")
    ap.add_argument("--languages", nargs="*", default=CODES)
    ap.add_argument("--final", default="3,32", help="the runtime's final limit a,b (DecodeLimits)")
    ap.add_argument("--draft", default="2.5,24", help="the runtime's draft limit a,b")
    ap.add_argument("--retry-penalty", type=float, default=1.2)
    ap.add_argument("--resume", action="store_true", help="keep the languages already in --json (a run cut short)")
    args = ap.parse_args()

    nllb = Nllb(args.model, args.device, args.beam, args.prefix)
    rows: dict[str, list] = {"final": [], "draft": []}  # [code, sentence, src pieces, out pieces, ended, distinct4]
    loops: dict[str, list] = {"final": [], "draft": []}
    done: list[str] = []
    if args.resume and args.json and args.json.exists():
        saved = json.loads(args.json.read_text(encoding="utf-8"))
        rows, loops, done = saved["rows"], saved["loops"], saved.get("done", [])
        print(f"resuming after {len(done)} language(s)", flush=True)
    texts: dict[tuple[str, str], list[str]] = {}
    for code in args.languages:
        path = args.flores / "devtest" / f"{code}.devtest"
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        finals = [lines[i] for i in pick(lines, args.sample)]
        texts[(code, "final")] = finals
        texts[(code, "draft")] = [first_half(t, code) for t in finals]
        if code in done:
            continue
        for kind in ("final", "draft"):
            batch = texts[(code, kind)]
            for i, (text, (out, ended)) in enumerate(zip(batch, nllb.decode(code, batch, args.cap))):
                d4 = distinct4(out)
                rows[kind].append([code, i, len(nllb.pieces(text)), len(out), ended, round(d4, 3)])
                if not ended or d4 < 0.5:
                    loops[kind].append({"language": code, "sentence": i, "source": text[:160],
                                        "output": nllb.sp.decode(out)[:200], "pieces": len(out), "ended": ended})
        mine = [r for r in rows["final"] if r[0] == code]
        print(f"{code}: {len(mine)} finals, longest output {max(r[3] for r in mine)} pieces, "
              f"max ratio {max(r[3] / max(1, r[2]) for r in mine):.2f}, loops {sum(l['language'] == code for l in loops['final'])} "
              f"final / {sum(l['language'] == code for l in loops['draft'])} draft", flush=True)
        done.append(code)
        if args.json:  # after every language, so a run cut short can --resume
            args.json.write_text(json.dumps({"done": done, "loops": loops, "rows": rows}, ensure_ascii=False) + "\n",
                                 encoding="utf-8")

    summary: dict = {"setting": nllb.setting, "cap": args.cap, "sentences": "all" if args.sample <= 0 else args.sample}
    for kind, setting in (("final", args.final), ("draft", args.draft)):
        loop_keys = {(l["language"], l["sentence"]) for l in loops[kind]}
        others = [r for r in rows[kind] if (r[0], r[1]) not in loop_keys]
        ratios = sorted(r[3] / max(1, r[2]) for r in others)
        q = lambda p: ratios[min(len(ratios) - 1, int(p * len(ratios)))]  # noqa: E731
        s = {"decodes": len(rows[kind]), "loops": len(loop_keys), "never_ended": sum(not r[4] for r in rows[kind]),
             "others_longest_output": max(r[3] for r in others),
             "others_ratio": {"p50": round(q(0.5), 2), "p99": round(q(0.99), 2), "max": round(ratios[-1], 2)},
             "grid": []}
        print(f"== {kind}s: {s['decodes']} decodes, {s['loops']} loops ({s['never_ended']} never ended); the others: "
              f"output/source p50 {s['others_ratio']['p50']} p99 {s['others_ratio']['p99']} max {s['others_ratio']['max']}, "
              f"longest {s['others_longest_output']} pieces")
        for a in (2.0, 2.5, 3.0):
            for b in (16, 24, 32):
                cut = [r for r in rows[kind] if r[3] + 1 > runtime_limit(a, b, r[2])]
                spare = min(runtime_limit(a, b, r[2]) - (r[3] + 1) for r in others)
                other_cuts = [r for r in cut if (r[0], r[1]) not in loop_keys]
                s["grid"].append({"a": a, "b": b, "loops_cut": len(cut) - len(other_cuts), "others_cut": len(other_cuts),
                                  "least_spare": spare})
                print(f"   {a} x src + {b:2d}: cuts {len(cut) - len(other_cuts)} loop(s), {len(other_cuts)} other(s); "
                      f"least spare among the others {spare}")
        a, b = (float(x) for x in setting.split(","))
        cut = [r for r in rows[kind] if r[3] + 1 > runtime_limit(a, b, r[2])]
        s["runtime_limit"] = {"a": a, "b": b, "cut": len(cut),
                              "loops_cut": sum((r[0], r[1]) in loop_keys for r in cut),
                              "loops_not_cut": sorted(f"{c}#{i}" for c, i in loop_keys - {(r[0], r[1]) for r in cut})}
        print(f"   runtime limit {a} x src + {b:g}: cuts {len(cut)}, of them {s['runtime_limit']['loops_cut']} loops; "
              f"loops that ended under it: {s['runtime_limit']['loops_not_cut']}")
        if kind == "final" and args.retry_penalty > 1.0:
            retries = []
            for r in cut:  # as the runtime retries: at the same limit, with the penalty
                code, i, src = r[0], r[1], r[2]
                cap = 1 + runtime_limit(a, b, src)
                (out, ended), = nllb.decode(code, [texts[(code, kind)][i]], cap, args.retry_penalty)
                retries.append({"language": code, "sentence": i, "ended": ended, "pieces": len(out),
                                "distinct4": round(distinct4(out), 3), "output": nllb.sp.decode(out)[:200]})
                print(f"   retry {code}#{i}: ended {ended}, {len(out)} pieces, distinct4 {distinct4(out):.2f}: "
                      f"{nllb.sp.decode(out)[:120]}")
            s["retries"] = retries
            s["retries_ended"] = sum(x["ended"] for x in retries)
        summary[kind] = s

    if args.json:
        args.json.write_text(json.dumps({"summary": summary, "done": done, "loops": loops, "rows": rows},
                                        ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
