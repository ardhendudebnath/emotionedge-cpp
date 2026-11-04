#!/usr/bin/env python3
"""Load test for `emotionedge serve`: N clients at once, each streaming a WAV file at speaking pace
over the WebSocket API, for each N given, and what they saw, summed up per N.

    python bench/load_server.py jfk.wav --url ws://127.0.0.1:8080/ --clients 1 3 8 16 \\
        [--start sync | --start stagger --stagger-ms 700] [--rounds 2] [--server-log serve.log] \\
        [--gpu] [--json load.json]

Start the server with --max-sessions at least the largest N, and --warm as many to keep model
loading out of the figures (or fewer, to measure refusals). Each client measures, as
apps/web/stream_wav.py does:
- first audio: from the end of an utterance (its transcript's `end`, at the pace sent) to the first
  translated audio after its translation, when no earlier translation was still playing;
- final caption: from the end of an utterance to its final translation;
- caption gap: the longest wait between an utterance's caption updates (drafts, then the final);
- the server's own end to end (from the session export), and whether it was refused.
Per N and round: p50/p95 over every client's utterances, utterances translated per second while
the clients ran, per-stage latency (median client) and each stage's busy time summed over every
session (its process() calls: a per-model profile under load). With --server-log (a
`serve --log debug` log), how long NLLB requests waited for their batch and ran, from the lines
logged during that run. With --gpu, peak GPU memory and mean utilization while it ran, and GPU
memory just before (nvidia-smi).
Requires `pip install websockets`.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
import statistics
import time
import wave
from pathlib import Path

NLLB_LINE = re.compile(
    r"translate: (draft|final)( retry)? waited ([0-9.]+) ms, ran ([0-9.]+) ms in a batch of (\d+) \((\d+) draft")


def load_wav(path: Path) -> tuple[int, bytes]:
    with wave.open(str(path)) as w:
        if w.getsampwidth() != 2:
            raise SystemExit(f"{path}: the input must be 16-bit PCM")
        rate, channels = w.getframerate(), w.getnchannels()
        frames = w.readframes(w.getnframes())
    if channels > 1:  # keep the first channel
        width = 2 * channels
        frames = b"".join(frames[i:i + 2] for i in range(0, len(frames), width))
    return rate, frames


async def client(index: int, delay: float, rate: int, frames: bytes, url: str, timeout: float) -> dict:
    """One session: streams at speaking pace, and returns what it measured."""
    import websockets  # type: ignore
    from websockets.exceptions import WebSocketException  # type: ignore

    await asyncio.sleep(delay)
    out: dict = {"client": index, "refused": None, "error": None, "first_audio_ms": [], "final_ms": [],
                 "caption_gap_ms": [], "utterances": 0, "server_e2e": None, "truncated": 0}
    events: list[dict] = []
    first_audio_after: dict[int, float] = {}
    url = url + ("&" if "?" in url else "?") + f"rate={rate}"
    try:
        async with websockets.connect(url, max_size=None, open_timeout=timeout) as ws:
            ready = json.loads(await asyncio.wait_for(ws.recv(), timeout))
            if ready.get("type") != "ready":
                out["refused"] = ready.get("message", str(ready))
                return out
            start = time.monotonic()
            out["started"] = time.time()

            async def send() -> None:
                block = rate // 50 * 2  # 20 ms of 16-bit samples
                for i, pos in enumerate(range(0, len(frames), block)):
                    await asyncio.sleep(max(0.0, start + i * 0.02 - time.monotonic()))
                    await ws.send(frames[pos:pos + block])
                await ws.send(json.dumps({"type": "end"}))

            async def receive() -> None:
                pending: list[int] = []  # translated while no audio was playing
                last_audio = -1.0
                async for message in ws:
                    now = time.monotonic() - start
                    if isinstance(message, bytes):
                        last_audio = now
                        for utt in pending:
                            first_audio_after[utt] = now
                        pending.clear()
                        continue
                    e = json.loads(message)
                    e["_t"] = now
                    events.append(e)
                    if e["type"] == "translation" and e["final"] and now - last_audio > 0.15:
                        pending.append(e["utterance"])  # audio arrives every 10 ms while one plays
                    elif e["type"] == "error":
                        out["error"] = e.get("message", "error")
                    elif e["type"] == "done":
                        return

            await asyncio.wait_for(asyncio.gather(send(), receive()), timeout)
            out["finished"] = time.time()
    except (OSError, asyncio.TimeoutError, WebSocketException) as e:
        out["refused"] = out["refused"] or f"{type(e).__name__}: {e}"
        return out

    ends = {e["utterance"]: e["end"] for e in events if e["type"] == "transcript" and e["final"]}
    for e in events:
        if e["type"] == "translation" and e["final"] and e["utterance"] in ends:
            out["utterances"] += 1
            out["final_ms"].append(1000.0 * (e["_t"] - ends[e["utterance"]]))
            out["truncated"] += bool(e.get("truncated"))
    out["first_audio_ms"] = [1000.0 * (t - ends[u]) for u, t in first_audio_after.items() if u in ends]
    by_utt: dict[int, list[dict]] = {}
    for e in events:
        if e["type"] in ("transcript", "translation") and "utterance" in e:
            by_utt.setdefault(e["utterance"], []).append(e)
    for evs in by_utt.values():
        heard = next((x["_t"] for x in evs if x["type"] == "transcript"), None)
        final = next((x["_t"] for x in evs if x["type"] == "translation" and x["final"]), None)
        if heard is None or final is None or final < heard:
            continue
        times = [heard] + sorted(x["_t"] for x in evs if x["type"] == "translation" and not x["final"]
                                 and heard <= x["_t"] <= final) + [final]
        out["caption_gap_ms"].append(1000.0 * max(b - a for a, b in zip(times, times[1:])))
    done = next((e for e in events if e["type"] == "done"), None)
    if done and done["session"].get("end_to_end", {}).get("count"):
        out["server_e2e"] = done["session"]["end_to_end"]
    if done:  # the session's latency budget (p50 ms per row) and each stage's busy time
        out["budget_p50_ms"] = {r["row"]: r["p50_ms"] for r in done["session"].get("latency_budget", []) if r.get("count")}
        out["stage_time"] = done["session"].get("stage_time", {})
    return out


class GpuSampler:
    """nvidia-smi's used memory (MiB) and utilization (%), every 250 ms while running."""

    def __init__(self) -> None:
        self.samples: list[tuple[float, float]] = []
        self.proc: asyncio.subprocess.Process | None = None
        self.reader: asyncio.Task | None = None

    @staticmethod
    async def once() -> float | None:
        if shutil.which("nvidia-smi") is None:
            return None
        proc = await asyncio.create_subprocess_exec("nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                                                    stdout=asyncio.subprocess.PIPE)
        stdout, _ = await proc.communicate()
        try:
            return float(stdout.decode().split()[0])
        except (IndexError, ValueError):
            return None

    async def start(self) -> None:
        if shutil.which("nvidia-smi") is None:
            return
        self.proc = await asyncio.create_subprocess_exec(
            "nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits", "-lms", "250",
            stdout=asyncio.subprocess.PIPE)

        async def read() -> None:
            assert self.proc and self.proc.stdout
            async for line in self.proc.stdout:
                try:
                    mem, util = (float(x) for x in line.decode().split(","))
                    self.samples.append((mem, util))
                except ValueError:
                    pass

        self.reader = asyncio.create_task(read())

    async def stop(self) -> dict:
        if self.proc is None:
            return {}
        self.proc.terminate()
        await self.proc.wait()
        if self.reader:
            await self.reader
        if not self.samples:
            return {}
        return {"peak_mib": max(m for m, _ in self.samples),
                "mean_util_pct": round(statistics.fmean(u for _, u in self.samples), 1)}


def pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, int(p * len(values)))]


def nllb_summary(log_text: str) -> dict:
    """Per kind: wait and run percentiles, mean batch size and drafts per batch; and every request
    in log order as [kind, waited ms, ran ms, batch size, drafts in the batch]."""
    rows: dict[str, list[tuple[float, float, int, int]]] = {"draft": [], "final": []}
    requests = []
    for m in NLLB_LINE.finditer(log_text):
        r = (float(m.group(3)), float(m.group(4)), int(m.group(5)), int(m.group(6)))
        rows[m.group(1)].append(r)
        requests.append([m.group(1), *r])
    out: dict = {"requests": requests}
    for kind, r in rows.items():
        if r:
            out[kind] = {"n": len(r), "wait_p50_ms": pct([x[0] for x in r], 0.5), "wait_p95_ms": pct([x[0] for x in r], 0.95),
                         "run_p50_ms": pct([x[1] for x in r], 0.5), "run_p95_ms": pct([x[1] for x in r], 0.95),
                         "mean_batch": round(statistics.fmean(x[2] for x in r), 2),
                         "mean_drafts_in_batch": round(statistics.fmean(x[3] for x in r), 2)}
    return out


async def run_once(n: int, args, rate: int, frames: bytes) -> dict:
    log_from = args.server_log.stat().st_size if args.server_log and args.server_log.exists() else 0
    sampler = GpuSampler()
    loaded = await GpuSampler.once() if args.gpu else None
    if args.gpu:
        await sampler.start()
    delays = [0.0 if args.start == "sync" else i * args.stagger_ms / 1000.0 for i in range(n)]
    began = time.time()
    clients = await asyncio.gather(*(client(i, d, rate, frames, args.url, args.timeout) for i, d in enumerate(delays)))
    wall = time.time() - began
    gpu = await sampler.stop() if args.gpu else {}
    served = [c for c in clients if c["refused"] is None]
    span = (max(c["finished"] for c in served) - min(c["started"] for c in served)) if served else 0.0
    row = {"clients": n, "start": args.start, "refused": n - len(served),
           "refusals": sorted({c["refused"] for c in clients if c["refused"]}), "wall_s": round(wall, 2),
           "utterances": sum(c["utterances"] for c in served),
           "utterances_per_s": round(sum(c["utterances"] for c in served) / span, 3) if span > 0 else None,
           "errors": sum(c["error"] is not None for c in served), "truncated": sum(c["truncated"] for c in served)}
    for key in ("first_audio_ms", "final_ms", "caption_gap_ms"):
        values = [v for c in served for v in c[key]]
        row[key] = {"n": len(values), "p50": pct(values, 0.5), "p95": pct(values, 0.95)}
    e2e = [c["server_e2e"]["p50_ms"] for c in served if c["server_e2e"]]
    row["server_e2e_p50_ms_median_client"] = statistics.median(e2e) if e2e else None
    stages: dict[str, list[float]] = {}
    for c in served:
        for name, ms in c.get("budget_p50_ms", {}).items():
            stages.setdefault(name, []).append(ms)
    row["stage_p50_ms_median_client"] = {k: round(statistics.median(v), 1) for k, v in stages.items()}
    busy: dict[str, list[float]] = {}  # each stage's process() time and calls, summed over every session
    for c in served:
        for name, t in c.get("stage_time", {}).items():
            busy.setdefault(name, [0.0, 0.0])
            busy[name][0] += t["busy_ms"]
            busy[name][1] += t["calls"]
    row["stage_busy_ms"] = {k: {"total": round(v[0]), "per_call": round(v[0] / v[1], 2) if v[1] else None}
                            for k, v in sorted(busy.items(), key=lambda kv: -kv[1][0])}
    if args.gpu:
        row["gpu"] = {"loaded_mib": loaded, **gpu}
    if args.server_log and args.server_log.exists():
        with args.server_log.open("rb") as f:
            f.seek(log_from)
            row["nllb"] = nllb_summary(f.read().decode("utf-8", errors="replace"))
    row["per_client"] = clients
    return row


def fmt(x: float | None) -> str:
    return "-" if x is None else f"{x:.0f}"


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wav", type=Path)
    ap.add_argument("--url", default="ws://127.0.0.1:8080/")
    ap.add_argument("--clients", type=int, nargs="+", default=[1, 3, 8, 16])
    ap.add_argument("--start", choices=["sync", "stagger"], default="sync",
                    help="sync: every client at once; stagger: one every --stagger-ms")
    ap.add_argument("--stagger-ms", type=float, default=700.0)
    ap.add_argument("--rounds", type=int, default=1, help="runs per N")
    ap.add_argument("--pause", type=float, default=8.0, help="seconds between runs")
    ap.add_argument("--timeout", type=float, default=300.0, help="per client, seconds")
    ap.add_argument("--server-log", type=Path, help="the server's --log debug output, for NLLB wait and run times")
    ap.add_argument("--gpu", action="store_true", help="sample nvidia-smi (the server on this machine)")
    ap.add_argument("--json", type=Path, help="every run, every client")
    args = ap.parse_args()

    rate, frames = load_wav(args.wav)
    rows = []
    print("clients start   round refused  utt/s  first audio p50/p95  final caption p50/p95  caption gap p95  "
          "NLLB final wait p50/p95 run p50  GPU MiB loaded/peak util%")
    for n in args.clients:
        for r in range(args.rounds):
            row = await run_once(n, args, rate, frames)
            row["round"] = r + 1
            rows.append(row)
            nf = row.get("nllb", {}).get("final", {})
            g = row.get("gpu", {})
            print(f"{n:7d} {args.start:7s} {r + 1:5d} {row['refused']:7d} {row['utterances_per_s'] or 0:6.2f}  "
                  f"{fmt(row['first_audio_ms']['p50']):>9s}/{fmt(row['first_audio_ms']['p95']):<9s}  "
                  f"{fmt(row['final_ms']['p50']):>11s}/{fmt(row['final_ms']['p95']):<10s}  "
                  f"{fmt(row['caption_gap_ms']['p95']):>15s}  "
                  f"{fmt(nf.get('wait_p50_ms')):>15s}/{fmt(nf.get('wait_p95_ms')):<5s} {fmt(nf.get('run_p50_ms')):>7s}  "
                  f"{fmt(g.get('loaded_mib')):>10s}/{fmt(g.get('peak_mib')):<6s} {g.get('mean_util_pct', '-')}", flush=True)
            if row["stage_p50_ms_median_client"]:
                print("        stage p50 ms (median client): " + ", ".join(
                    f"{k} {v:.0f}" for k, v in row["stage_p50_ms_median_client"].items()), flush=True)
            if row["stage_busy_ms"]:
                print("        busy ms, all sessions (per call): " + ", ".join(
                    f"{k} {v['total']} ({v['per_call']})" for k, v in list(row["stage_busy_ms"].items())[:6]), flush=True)
            if args.json:
                args.json.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
            await asyncio.sleep(args.pause)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
