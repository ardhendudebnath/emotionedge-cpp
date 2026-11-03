#!/usr/bin/env python3
"""P4 data: emotion-faithful Hindi targets for the LoRA corpus from a local LLM. The P2 teacher
(ml/data/teacher_translate.py) reads the plain English and never sees the emotion, so its targets
can carry little of it; here the LLM is told each item's emotion, arousal and register.

    llama-server -m Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf -ngl 99 -c 16384 -np 8 --port 8090 &
    python ml/data/llm_translate.py --corpus data/mt_corpus.hi.jsonl --out data/mt_corpus.llm.jsonl \\
        [--mode translate | --mode postedit] [--field tgt_llm] [--split valid --limit 300]

--mode translate translates the English; --mode postedit edits the teacher's Hindi ("tgt") toward
the emotion with the smallest changes, so its meaning stays the teacher's. Adds --field to every
line (null when the reply is not a usable Hindi line, with the reason in "<field>_reject") and
resumes from --out. Talks to any OpenAI-compatible chat endpoint (--url); no paid API is needed
or used. Standard library only.
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# A first prompt also listed Hindi particles (अरे, यार, ...) as ways to carry the emotion: the model
# then put "अरे" into most lines, calm and neutral ones too, and the round trip lost meaning
# (chrF 62.0 -> 44.2 on 300 held-out items). Both prompts now forbid adding anything.
SYSTEM = {
    "translate": (
        "You translate English dialogue into Hindi for dubbing. Translate faithfully: keep every piece of meaning "
        "and add nothing, in particular no interjection, particle or exclamation the English does not have. Among "
        "equally faithful wordings, choose the one a native Hindi speaker feeling the given emotion would say, and "
        "let the punctuation carry it too. Write Devanagari; keep names and numbers, and the English words a Hindi "
        "speaker would keep. Answer with the Hindi line only."),
    "postedit": (
        "You edit Hindi dubbing lines. You get an English line, the emotion it is spoken with, and a faithful Hindi "
        "translation of it. Return the Hindi with the smallest changes that make it sound the way a native speaker "
        "feeling that emotion would say it: word choice, word order, punctuation. Keep the meaning exactly; add no "
        "interjection, particle or exclamation the English does not have. If it already fits, return it unchanged. "
        "Answer with the Hindi line only."),
}
REGISTER = {"casual": "casual (तुम)", "formal": "formal (आप)", "neutral": "neutral"}
DEVANAGARI = re.compile(r"[ऀ-ॿ]")
LATIN = re.compile(r"[A-Za-z]")
LABEL = re.compile(r"^\s*(hindi|translation|अनुवाद)\s*[:：]\s*", re.I)


def prompt(item: dict, mode: str) -> str:
    text = (f"Emotion: {item['emotion']} (arousal {item['arousal']:+.1f} on a scale from -1 to 1). "
            f"Register: {REGISTER.get(item['register'], item['register'])}.\nEnglish: {item['src']}")
    return text + (f"\nHindi: {item['tgt']}" if mode == "postedit" else "")


def check(reply: str, item: dict) -> tuple[str | None, str]:
    """The Hindi line, or None and why not."""
    lines = [line.strip() for line in reply.strip().splitlines() if line.strip()]
    if not lines:
        return None, "empty"
    text = LABEL.sub("", lines[0]).strip().strip('"').strip()
    deva, latin = len(DEVANAGARI.findall(text)), len(LATIN.findall(text))
    if deva == 0 or deva < 2 * latin:
        return None, "not Hindi"
    if len(text) > 4 * max(len(item.get("tgt") or ""), len(item["src"])) + 20:
        return None, "too long"
    return text, ""


def ask(url: str, model: str, item: dict, mode: str, field: str, temperature: float, timeout: float) -> dict:
    body = {"model": model, "temperature": temperature, "top_p": 0.9,
            "max_tokens": 3 * len(item["src"].split()) + 48,
            "messages": [{"role": "system", "content": SYSTEM[mode]}, {"role": "user", "content": prompt(item, mode)}]}
    request = urllib.request.Request(url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                reply = json.loads(response.read())["choices"][0]["message"]["content"]
            text, reject = check(reply, item)
            return {**item, field: text, **({field + "_reject": reject} if reject else {})}
        except (OSError, KeyError, ValueError) as e:
            if attempt == 2:
                return {**item, field: None, field + "_reject": f"request failed: {e}"}
            time.sleep(2.0 * (attempt + 1))
    raise AssertionError("unreachable")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, required=True, help="JSONL from build_mt_corpus.py (+ teacher tgt)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8090/v1")
    parser.add_argument("--model", default="local")
    parser.add_argument("--mode", choices=sorted(SYSTEM), default="translate",
                        help="translate the English, or post-edit the teacher's Hindi (`tgt`) toward the emotion")
    parser.add_argument("--field", default="tgt_llm", help="the field the Hindi goes into")
    parser.add_argument("--split", help="only this split (e.g. valid, for a pilot)")
    parser.add_argument("--limit", type=int, default=0, help="at most this many items (spread over the corpus)")
    parser.add_argument("--parallel", type=int, default=8, help="requests in flight (the server's -np)")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    items = [json.loads(line) for line in args.corpus.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.split:
        items = [i for i in items if i.get("split") == args.split]
    if args.limit and len(items) > args.limit:
        step = len(items) / args.limit
        items = [items[int(k * step)] for k in range(args.limit)]
    done = set()
    if args.out.exists():
        done = {json.loads(line)["id"] for line in args.out.read_text(encoding="utf-8").splitlines() if line.strip()}
    todo = [i for i in items if i["id"] not in done]
    print(f"{len(items)} items, {len(done)} already done, {len(todo)} to go", flush=True)

    start, written, rejected = time.time(), 0, 0
    with open(args.out, "a", encoding="utf-8") as out, ThreadPoolExecutor(args.parallel) as pool:
        for row in pool.map(lambda i: ask(args.url, args.model, i, args.mode, args.field, args.temperature, args.timeout),
                            todo):
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
            rejected += row[args.field] is None
            if written % 200 == 0:
                print(f"  {written}/{len(todo)}  {written / (time.time() - start):.1f} items/s, {rejected} rejected", flush=True)
    print(f"done: {written} in {time.time() - start:.0f} s, {rejected} rejected -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
