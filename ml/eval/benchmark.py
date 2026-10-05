#!/usr/bin/env python3
"""P5 public benchmark: re-measures, in one command, the numbers the README quotes, and writes
them up with the machine, the models and the spread over repeated runs.

    python ml/eval/benchmark.py run --emotionedge build/engines/apps/emotionedge \\
        --manifest models/manifest.json --data data --espeak-data /usr/lib/x86_64-linux-gnu \\
        --out out/benchmark --runs 5 --devices cpu auto
    python ml/eval/benchmark.py report --results out/benchmark/results.json --out docs/benchmark.md

Suites (--suites, default all; results.json keeps the others, so suites can run separately):
- latency: real-time runs, per device, of whisper.cpp's jfk.wav and of three back-to-back
  sentences spoken by Piper (`emotionedge say`). Latency budget rows, end to end, ASR WER,
  peak RAM, CPU time and GPU memory.
- transfer: offline runs of the 48-clip RAVDESS input (eval_ecs.py build). ECS and the transfer
  metrics (output label agreement, non-neutral share, arousal CCC), and ASR WER.
- emotion: the emotion stage alone (`emotionedge emotion`). Voice on RAVDESS (1440 clips),
  words on the MELD test split.
- mt: FLORES-200 devtest eng->hin with eval_mt.py, per device. It uses CTranslate2's Python API
  on the runtime's model files, prefix and settings.
- overhead: ee_bench on the stand-in engines (--ee-bench): the runtime's own cost.

--data holds ravdess.jsonl and meld_test.jsonl (eval_emotion.py prepare-ravdess / prepare-meld)
and flores200_dataset/. Standard library only, except the mt suite and building the RAVDESS input,
which run with --python (ctranslate2, sentencepiece, sacrebleu; numpy, soundfile). Linux measures
peak RAM and CPU time per run; nvidia-smi, if present, the GPU memory.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path
from typing import Dict, List, Optional, Sequence

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "ml"))
from emotionedge_ml.metrics import wer  # noqa: E402

SUITES = ("latency", "transfer", "emotion", "mt", "overhead")
JFK_TEXT = "And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country."
SENTENCES = ["I can't believe you did this!", "I'm so sorry, I didn't mean to hurt you.",
             "This is amazing, thank you so much!"]
SENTENCES_VOICE = "tts.piper.en_US.lessac.medium"
RAVDESS_STATEMENTS = {"01": "Kids are talking by the door.", "02": "Dogs are sitting by the door."}
TARGETS = {"end_to_end_p95_ms": 800.0, "asr_rtf_p95": 0.3, "ecs": 0.75, "peak_rss_mb": 3072.0}


# ---------------------------------------------------------------------------------------- runs

class GpuSampler:
    """Peak GPU memory in use while a run lasts, less what was in use before it (nvidia-smi)."""

    def __init__(self) -> None:
        self.tool = shutil.which("nvidia-smi")
        self.peak: Optional[float] = None
        self.proc: Optional[subprocess.Popen] = None
        self.thread: Optional[threading.Thread] = None

    def _query(self) -> Optional[float]:
        out = subprocess.run([self.tool, "-i", "0", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True)
        return float(out.stdout.strip()) if out.returncode == 0 and out.stdout.strip() else None

    def start(self) -> None:
        if not self.tool:
            return
        self.before = self._query()
        if self.before is None:
            return
        self.proc = subprocess.Popen([self.tool, "-i", "0", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                                      "-lms", "250"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self) -> None:
        for line in self.proc.stdout:
            try:
                value = float(line.strip())
            except ValueError:
                continue
            self.peak = value if self.peak is None else max(self.peak, value)

    def stop(self) -> Optional[float]:
        if self.proc is None:
            return None
        self.proc.terminate()
        self.proc.wait()
        self.thread.join(timeout=2)
        return None if self.peak is None else max(0.0, self.peak - self.before)


def measure(cmd: Sequence[str], log: Path) -> Dict[str, Optional[float]]:
    """Runs a command to completion: wall time, CPU time and peak RAM of the process (Linux),
    and GPU memory. Raises with the log's tail when it fails."""
    log.parent.mkdir(parents=True, exist_ok=True)
    gpu = GpuSampler()
    gpu.start()
    start = time.monotonic()
    with log.open("w", encoding="utf-8") as f:
        proc = subprocess.Popen([str(c) for c in cmd], cwd=REPO, stdout=f, stderr=subprocess.STDOUT)
        usage = None
        if hasattr(os, "wait4"):
            _, status, usage = os.wait4(proc.pid, 0)
            proc.returncode = os.waitstatus_to_exitcode(status)
        else:
            proc.wait()
    wall = time.monotonic() - start
    gpu_mb = gpu.stop()
    if proc.returncode != 0:
        tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]
        raise RuntimeError(f"{Path(cmd[0]).name} {cmd[1]} failed ({proc.returncode}), {log}:\n" + "\n".join(tail))
    return {"wall_s": wall,
            "cpu_s": None if usage is None else usage.ru_utime + usage.ru_stime,
            "peak_rss_mb": None if usage is None else usage.ru_maxrss / 1024.0,
            "gpu_mem_mb": gpu_mb}


def conditions() -> Dict[str, Optional[float]]:
    """The machine's state as a run starts: load average and the GPU's temperature and clock.
    On a laptop these move real-time latency by up to 2x (a run right after a long compile).
    The clock stays out of the report: under WSL, nvidia-smi sometimes reads it as 7192 MHz."""
    out: Dict[str, Optional[float]] = {"load_1m": None, "gpu_temp_c": None, "gpu_sm_mhz": None}
    if Path("/proc/loadavg").exists():
        out["load_1m"] = float(Path("/proc/loadavg").read_text().split()[0])
    if shutil.which("nvidia-smi"):
        line = first_line(["nvidia-smi", "-i", "0", "--query-gpu=temperature.gpu,clocks.sm", "--format=csv,noheader,nounits"])
        try:
            out["gpu_temp_c"], out["gpu_sm_mhz"] = (float(x) for x in line.split(","))
        except ValueError:
            pass
    return out


def wav_seconds(path: Path) -> float:
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


def ee_sets(args, device: Optional[str]) -> List[str]:
    sets = ["--set", f"pipeline.models={args.manifest}"]
    if args.espeak_data:
        sets += ["--set", f"tts.espeak_data={args.espeak_data}"]
    if device:
        sets += ["--set", f"pipeline.device={device}"]
    for kv in args.set:
        sets += ["--set", kv]
    return sets


def ee_run(args, wav: Path, out: Path, device: str, realtime: bool) -> dict:
    cmd = [args.emotionedge, "run", "--input", wav, "--config", args.config, "--out", out, *ee_sets(args, device)]
    if realtime:
        cmd.append("--realtime")
    before = conditions()
    stats = measure(cmd, out.with_suffix(".log"))
    session = json.loads((out / "session.json").read_text(encoding="utf-8"))
    rows = {r["row"]: {"p50": r["p50_ms"], "p95": r["p95_ms"], "target": r["target_ms"], "count": r["count"]}
            for r in session["latency_budget"]}
    prom = (out / "metrics.prom").read_text(encoding="utf-8") if (out / "metrics.prom").exists() else ""
    dropouts = re.search(r"^ee_audio_dropout_samples_total (\S+)", prom, re.M)
    e2e = session["end_to_end"]
    return {**stats, **before, "audio_s": wav_seconds(wav), "rows": rows,
            "end_to_end_p50_ms": e2e["p50_ms"] if e2e["count"] else None,
            "end_to_end_p95_ms": e2e["p95_ms"] if e2e["count"] else None,
            "asr_rtf_p95": session["asr_rtf"]["p95"], "dropout_samples": float(dropouts.group(1)) if dropouts else None,
            "utterances": session["utterances"]}


def transcript(utterances: List[dict]) -> str:
    return " ".join(u["source"]["text"] for u in utterances)


def ravdess_wer(utterances: List[dict], clips: List[dict]) -> float:
    """WER over the clips: each clip's statement against the utterances whose middle falls in it."""
    refs, hyps = [], []
    for c in clips:
        refs.append(RAVDESS_STATEMENTS[c["id"].split("-")[4]])
        hyps.append(" ".join(u["source"]["text"] for u in utterances
                             if c["start"] <= 0.5 * (u["source"]["start"] + u["source"]["end"]) <= c["end"]))
    return wer(refs, hyps)


def scored(script: str, python: str, argv: List[str], name: str, out: Path) -> Dict[str, float]:
    """Runs one of the eval scripts with --name/--json and returns its metrics, suffix removed."""
    path = out.with_suffix(".score.json")
    path.unlink(missing_ok=True)
    measure([python, REPO / "ml" / "eval" / script, *argv, "--name", name, "--json", path], out.with_suffix(".score.log"))
    suffix = "_" + name
    return {k[:-len(suffix)]: v for k, v in json.loads(path.read_text(encoding="utf-8")).items() if k.endswith(suffix)}


# -------------------------------------------------------------------------------------- suites

def prepare_inputs(args) -> Dict[str, Path]:
    inputs = {}
    jfk = args.jfk or next(iter(sorted(Path(args.emotionedge).resolve().parents[1].glob("_deps/whisper-src/samples/jfk.wav"))), None)
    if jfk is None or not Path(jfk).exists():
        raise SystemExit("jfk.wav not found: give --jfk (whisper.cpp's samples/jfk.wav)")
    inputs["jfk"] = Path(jfk)
    sentences = args.out / "inputs" / "sentences.wav"
    if not sentences.exists():
        cmd = [args.emotionedge, "say", "--engine", "piper", "--model-id", SENTENCES_VOICE, "--manifest", args.manifest,
               "--text", " | ".join(SENTENCES), "--out", sentences]
        if args.espeak_data:
            cmd += ["--espeak-data", args.espeak_data]
        measure(cmd, sentences.with_suffix(".log"))
    inputs["sentences"] = sentences
    return inputs


def latency_suite(args, results: dict) -> None:
    inputs = prepare_inputs(args)
    references = {"jfk": JFK_TEXT, "sentences": " ".join(SENTENCES)}
    suite = results["latency"] = {}
    for device in args.devices:
        for i in range(args.warmup):  # loads caches and the GPU's clocks; not recorded
            print(f"latency: {device} warm-up {i + 1}/{args.warmup}", flush=True)
            ee_run(args, inputs["jfk"], args.out / "runs" / f"latency-{device}-warmup", device, realtime=True)
        for name, wav in inputs.items():
            runs = suite.setdefault(device, {}).setdefault(name, [])
            for i in range(args.runs):
                print(f"latency: {device} {name} run {i + 1}/{args.runs}", flush=True)
                r = ee_run(args, wav, args.out / "runs" / f"latency-{device}-{name}-{i + 1}", device, realtime=True)
                text = transcript(r.pop("utterances"))
                runs.append({**r, "wer": wer([references[name]], [text]), "transcript": text})


def ravdess_input(args) -> Dict[str, Path]:
    wav, clips = args.data / "ravdess_ecs.wav", args.data / "ravdess_ecs.json"
    if not wav.exists():
        measure([args.python, REPO / "ml" / "eval" / "eval_ecs.py", "build", "--items", args.data / "ravdess.jsonl",
                 "--out", wav], args.out / "inputs" / "ravdess_ecs.log")
    return {"wav": wav, "clips": clips}


def transfer_suite(args, results: dict) -> None:
    ravdess = ravdess_input(args)
    clips = json.loads(ravdess["clips"].read_text(encoding="utf-8"))
    runs = results["transfer"] = {"device": args.quality_device, "clips": len(clips), "runs": []}
    for i in range(args.runs):
        print(f"transfer: run {i + 1}/{args.runs}", flush=True)
        out = args.out / "runs" / f"transfer-{i + 1}"
        r = ee_run(args, ravdess["wav"], out, args.quality_device, realtime=False)
        metrics = scored("eval_ecs.py", sys.executable,
                         ["score", "--clips", ravdess["clips"], "--session", out / "session.json"], "run", out)
        runs["runs"].append({"wer": ravdess_wer(r.pop("utterances"), clips), "peak_rss_mb": r["peak_rss_mb"],
                             "gpu_mem_mb": r["gpu_mem_mb"], **metrics})


def emotion_suite(args, results: dict) -> None:
    suite = results["emotion"] = {"device": args.quality_device}
    for name, items, modality, extra in (("ravdess", "ravdess.jsonl", "acoustic", ["--set", "emotion.lexical=none"]),
                                         ("meld", "meld_test.jsonl", "lexical", [])):
        print(f"emotion: {name}", flush=True)
        out = args.out / "runs" / f"emotion-{name}"
        preds = out.with_suffix(".jsonl")
        stats = measure([args.emotionedge, "emotion", "--config", args.config, "--manifest", args.data / items,
                         "--out", preds, *ee_sets(args, args.quality_device), *extra], out.with_suffix(".run.log"))
        metrics = scored("eval_emotion.py", sys.executable,
                         ["score", "--items", args.data / items, "--preds", preds, "--modality", modality], name, out)
        suite[name] = {**metrics, "items": sum(1 for _ in preds.open(encoding="utf-8")), "wall_s": stats["wall_s"]}


def mt_suite(args, results: dict) -> None:
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    entry = next(m for m in manifest["models"] if m["id"] == args.mt_model_id)
    model = Path(args.manifest).parent / entry["path"]
    suite = results["mt"] = {"model": args.mt_model_id}
    for device in args.devices:
        print(f"mt: {device}", flush=True)
        metrics = scored("eval_mt.py", args.python,
                         ["--model", model, "--prefix", "runtime", "--flores", args.data / "flores200_dataset",
                          "--device", device], "run", args.out / "runs" / f"mt-{device}")
        suite[device] = {k.removesuffix("_flores"): v for k, v in metrics.items()}  # chrf_flores -> chrf


def overhead_suite(args, results: dict) -> None:
    if not args.ee_bench:
        print("overhead: skipped (no --ee-bench)")
        return
    runs = results["overhead"] = []
    for i in range(args.runs):
        print(f"overhead: run {i + 1}/{args.runs}", flush=True)
        out = args.out / "runs" / f"overhead-{i + 1}"
        measure([args.ee_bench, "--repeat", "3", "--out", out, "--json", out / "bench.json"], out.with_suffix(".log"))
        runs.append(json.loads((out / "bench.json").read_text(encoding="utf-8")))


# --------------------------------------------------------------------------------- environment

def first_line(cmd: Sequence[str]) -> str:
    try:
        out = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, cwd=REPO, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else ""


def environment(args) -> dict:
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True,
                            cwd=REPO).stdout.splitlines()
    env = {"date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
           "commit": first_line(["git", "rev-parse", "--short=12", "HEAD"]),
           "dirty": bool(status), "dirty_paths": [line[3:] for line in status],
           "emotionedge": first_line([args.emotionedge, "version"]),
           "os": platform.platform(), "cores": os.cpu_count(), "config": str(args.config)}
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        model = re.search(r"^model name\s*:\s*(.+)$", cpuinfo.read_text(), re.M)
        env["cpu"] = model.group(1).strip() if model else platform.processor()
        total = re.search(r"^MemTotal:\s*(\d+) kB", Path("/proc/meminfo").read_text(), re.M)
        env["ram_gb"] = round(int(total.group(1)) / 1048576, 1) if total else None
    else:
        env["cpu"] = platform.processor()
    if shutil.which("nvidia-smi"):
        env["gpu"] = first_line(["nvidia-smi", "-i", "0", "--query-gpu=name,driver_version,memory.total",
                                 "--format=csv,noheader"])
    # The models the config names, with the hashes the runtime verified them against.
    ids = re.findall(r"^\s*\w*model_id:\s*([\w.\-]+)", Path(REPO / args.config).read_text(encoding="utf-8"), re.M)
    manifest = {m["id"]: m for m in json.loads(Path(args.manifest).read_text(encoding="utf-8"))["models"]}
    env["models"] = [{"id": i, "precision": manifest[i].get("precision", ""), "sha256": manifest[i].get("sha256", "")[:12],
                      "license": manifest[i].get("license", "")} for i in dict.fromkeys(ids) if i in manifest]
    return env


# ------------------------------------------------------------------------------------- report

def span(values: Sequence[Optional[float]], digits: int = 0, scale: float = 1.0) -> str:
    """The range over runs, e.g. "365–406"; one value when they agree at this precision."""
    xs = [v * scale for v in values if v is not None]
    if not xs:
        return "–"
    lo, hi = f"{min(xs):.{digits}f}", f"{max(xs):.{digits}f}"
    return lo if lo == hi else f"{lo}–{hi}"


def spread(values: Sequence[Optional[float]], digits: int = 0) -> str:
    """The median and the range, e.g. "627 (614–647)"."""
    xs = sorted(v for v in values if v is not None)
    if len(xs) < 3:
        return span(xs, digits)
    mid = xs[len(xs) // 2] if len(xs) % 2 else 0.5 * (xs[len(xs) // 2 - 1] + xs[len(xs) // 2])
    return f"{mid:.{digits}f} ({span(xs, digits)})"


def worst(values: Sequence[Optional[float]], higher_is_better: bool = False) -> Optional[float]:
    xs = [v for v in values if v is not None]
    if not xs:
        return None
    return min(xs) if higher_is_better else max(xs)


def verdict(value: Optional[float], target: float, higher_is_better: bool = False) -> str:
    if value is None:
        return "–"
    return "met" if (value >= target if higher_is_better else value < target) else "**missed**"


def render(results: dict) -> str:
    env, settings = results.get("environment", {}), results.get("settings", {})
    latency, transfer = results.get("latency", {}), results.get("transfer", {})
    devices = list(latency)
    runs = settings.get("runs", "?")
    out = ["# EmotionEdge benchmark", "",
           f"Measured {env.get('date', '?')} at commit `{env.get('commit', '?')}`"
           + (f" with local changes to {', '.join(f'`{p}`' for p in env['dirty_paths'])}" if env.get("dirty_paths")
              else " with uncommitted changes" if env.get("dirty") else "") + ", translating English into Hindi with "
           f"`{env.get('config', '?')}`. Ranges are the lowest and highest of {runs} runs; where a median is "
           "given, it comes first. Every run is in `benchmark.json` next to this file. Regenerate both with "
           "`ml/eval/benchmark.py` (its docstring has the commands).", "",
           "| Machine | |", "|---|---|",
           f"| CPU | {env.get('cpu', '?')}, {env.get('cores', '?')} threads, {env.get('ram_gb', '?')} GB RAM |",
           f"| GPU | {env.get('gpu') or 'none'} |", f"| OS | {env.get('os', '?')} |",
           f"| Build | {env.get('emotionedge', '?')} |", ""]

    def lat(device: str, name: str, key: str) -> List[Optional[float]]:
        return [r.get(key) for r in latency.get(device, {}).get(name, [])]

    out += ["## Blueprint targets", "",
            "| Target | Goal | " + " | ".join(devices) + " |", "|---|---|" + "---|" * len(devices)]
    for label, name in (("End to end p95, jfk.wav (real time)", "jfk"),
                        ("End to end p95, 3 back-to-back sentences (real time)", "sentences")):
        cells = [f"{spread(lat(d, name, 'end_to_end_p95_ms'))} ms, "
                 f"{verdict(worst(lat(d, name, 'end_to_end_p95_ms')), TARGETS['end_to_end_p95_ms'])}" for d in devices]
        out.append(f"| {label} | < 800 ms | " + " | ".join(cells) + " |")
    cells = [f"{span(lat(d, 'jfk', 'asr_rtf_p95'), 3)}, {verdict(worst(lat(d, 'jfk', 'asr_rtf_p95')), 0.3)}"
             for d in devices]
    out.append("| ASR real-time factor p95, jfk.wav | < 0.3 | " + " | ".join(cells) + " |")
    rss = {d: [r.get("peak_rss_mb") for n in latency.get(d, {}).values() for r in n] for d in devices}
    cells = [f"{span(rss[d], 1, 1 / 1024)} GB, {verdict(worst(rss[d]), TARGETS['peak_rss_mb'])}" for d in devices]
    out.append("| Peak RAM, real-time runs | < 3 GB | " + " | ".join(cells) + " |")
    drops = {d: [r.get("dropout_samples") for n in latency.get(d, {}).values() for r in n] for d in devices}
    cells = [f"{span(drops[d])} samples, {verdict(worst(drops[d]), 0.5)}" for d in devices]
    out.append("| Audio dropouts, real-time runs | 0 | " + " | ".join(cells) + " |")
    if transfer.get("runs"):
        ecs = [r.get("ecs_mean") for r in transfer["runs"]]
        cell = f"{span(ecs, 3)}, {verdict(worst(ecs, True), TARGETS['ecs'], True)}"
        out.append("| Emotion consistency (ECS), RAVDESS | ≥ 0.75 | "
                   + " | ".join(cell if d == transfer.get("device") else "–" for d in devices) + " |")
    out.append("")

    for name, title in (("jfk", "jfk.wav: 11 s, 4 utterances with pauses"),
                        ("sentences", "3 back-to-back sentences: 10 s, Piper's voice")):
        out += [f"## Latency, real time: {title}", "",
                "| Row | Budget | " + " | ".join(f"{d} p50 | {d} p95" for d in devices) + " |",
                "|---|---|" + "---|---|" * len(devices)]
        rows = next((r["rows"] for d in devices for r in latency[d].get(name, [])), {})
        for row, info in rows.items():
            cells = []
            for d in devices:
                values = [r["rows"].get(row, {}) for r in latency[d].get(name, [])]
                cells += [span([v.get("p50") for v in values]), span([v.get("p95") for v in values])]
            out.append(f"| {row} | {info['target']:.0f} ms | " + " | ".join(cells) + " |")
        cells = []
        for d in devices:
            cells += [span(lat(d, name, "end_to_end_p50_ms")), span(lat(d, name, "end_to_end_p95_ms"))]
        out.append("| **End to end** | 800 ms (p95) | " + " | ".join(f"**{c}**" for c in cells) + " |")
        cells = []
        for d in devices:
            cpu = [r["cpu_s"] / r["audio_s"] for r in latency[d].get(name, []) if r.get("cpu_s") is not None]
            cells.append(f"CPU time {span(cpu, 1)} s per second of audio (model loading included), "
                         f"GPU memory {span(lat(d, name, 'gpu_mem_mb'))} MB, ASR WER {span(lat(d, name, 'wer'), 3)}. "
                         f"At the start of the runs: load average {span(lat(d, name, 'load_1m'), 1)}, "
                         f"GPU at {span(lat(d, name, 'gpu_temp_c'))} °C")
        out += ["", *[f"- **{d}:** {c}" for d, c in zip(devices, cells)], ""]

    out += ["## Quality", "", "| Measure | Data | Result |", "|---|---|---|"]
    if transfer.get("runs"):
        t = transfer["runs"]
        out += [f"| ASR WER | RAVDESS, {transfer['clips']} clips | {span([r['wer'] for r in t], 3)} |",
                f"| ECS, mean | RAVDESS, {transfer['clips']} clips | {span([r.get('ecs_mean') for r in t], 3)} |",
                f"| ECS ≥ 0.75, share of clauses | | {span([r.get('ecs_share_above_target') for r in t], 2)} |",
                f"| Input emotion read correctly | | {span([r.get('input_emotion_accuracy') for r in t], 2)} |",
                f"| Output heard as the input's emotion | | {span([r.get('output_label_agreement') for r in t], 2)} |",
                f"| Output heard as non-neutral | | {span([r.get('output_non_neutral_share') for r in t], 2)} |",
                f"| Arousal CCC, input vs output | | {span([r.get('arousal_ccc_in_out') for r in t], 2)} |"]
    emotion = results.get("emotion", {})
    for name, label, data in (("ravdess", "Emotion from the voice (emotion2vec+)", "RAVDESS"),
                              ("meld", "Emotion from the words (DistilRoBERTa)", "MELD test")):
        if name in emotion:
            e = emotion[name]
            out.append(f"| {label}: UAR / macro F1 | {data}, {e['items']} items | "
                       f"{e['emotion_uar']:.3f} / {e['emotion_f1']:.3f} |")
            out.append(f"| {label}: CCC V / A / D | | {e['ccc_valence']:.2f} / {e['ccc_arousal']:.2f} / "
                       f"{e['ccc_dominance']:.2f} |")
    mt = results.get("mt", {})
    for d in [k for k in mt if k != "model"]:
        out.append(f"| Translation chrF / BLEU ({d}) | FLORES-200 devtest eng→hin | "
                   f"{mt[d]['chrf']:.2f} / {mt[d]['bleu']:.2f} (prefix leaks {mt[d].get('leak_rate', 0):.3f}) |")
    out.append("")

    overhead = results.get("overhead", [])
    if overhead:
        out += ["## Runtime overhead (stand-in engines, `ee_bench`)", "",
                f"End to end p95 {span([b['end_to_end_ms']['p95'] for b in overhead], 1)} ms, peak RAM "
                f"{span([b['peak_rss_mb'] for b in overhead])} MB, dropouts "
                f"{span([b['dropout_samples'] for b in overhead])}: the scheduler, queues and DSP without "
                "neural models.", ""]

    models = env.get("models", [])
    if models:
        out += ["## Models", "", "| Model | Precision | SHA-256 | License |", "|---|---|---|---|"]
        out += [f"| `{m['id']}` | {m['precision']} | `{m['sha256'] or '–'}` | {m['license']} |" for m in models]
        out.append("")
    out += ["## How it was measured", "",
            "- **Real time** feeds the input at speaking pace through the threaded pipeline (`--realtime`), "
            "as a microphone would. End to end runs from the moment the speaker stops (the VAD endpoint) "
            "to the first translated audio.",
            "- **Devices:** `cpu` forces every model onto the CPU; `auto` puts whisper, NLLB, Kokoro, "
            "emotion2vec+ and ECAPA on the GPU when the build has CUDA.",
            "- **The machine's state matters.** On this laptop the same binary and config measured 1127–1294 ms "
            "p95 on jfk.wav in one session and 614–631 ms in another, taking 3–5× the CPU time for the same "
            "work. Each device gets an unrecorded warm-up run first, and every run logs the load average and "
            "the GPU's temperature as it starts. Keep the machine otherwise idle, and on mains power.",
            "- **Peak RAM and CPU time** are the process's own (`wait4`). GPU memory is the device's use during "
            "the run, less its use before (`nvidia-smi`), so other processes can inflate it.",
            "- **The emotion-transfer set** is 48 RAVDESS clips (4 actors × 8 emotions + neutral, one sentence). "
            "The emotion is in the voice only. One judge, emotion2vec+, reads both the input and the "
            "translated speech, so ECS is lenient: a neutral output still scores well. The label agreement, "
            "non-neutral share and arousal CCC show what actually transfers.",
            "- **Translation** is scored with CTranslate2's Python API on the runtime's model files, prefix "
            "and beam; the other numbers come from the C++ runtime.",
            "- **Small sets:** jfk.wav, three sentences and 48 clips. Between identical runs, per-emotion ECS "
            "moves by up to 0.024 and the arousal CCC by up to 0.05.", ""]
    return "\n".join(out)


# --------------------------------------------------------------------------------------- main

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run")
    p.add_argument("--emotionedge", required=True, help="the engines build's apps/emotionedge")
    p.add_argument("--manifest", type=Path, required=True, help="models/manifest.json next to the model files")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("out/benchmark"))
    p.add_argument("--config", default="config/pipeline.engines.yaml")
    p.add_argument("--espeak-data", help="espeak-ng-data's parent directory, if not the system's")
    p.add_argument("--jfk", type=Path, help="whisper.cpp's samples/jfk.wav (default: in the build tree)")
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1, help="unrecorded real-time runs per device first")
    p.add_argument("--devices", nargs="+", default=["cpu", "auto"])
    p.add_argument("--quality-device", default="auto", help="device for the transfer and emotion suites")
    p.add_argument("--suites", nargs="+", choices=SUITES, default=list(SUITES))
    p.add_argument("--python", default=sys.executable, help="Python with the mt suite's and eval_ecs build's packages")
    p.add_argument("--mt-model-id", default="mt.nllb200.distilled600m.emo.int8")
    p.add_argument("--ee-bench", help="ee_bench from a build with the stand-in engines")
    p.add_argument("--set", action="append", default=[], help="extra pipeline override, KEY=VALUE (repeatable)")
    p = sub.add_parser("report")
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "report":
        args.out.write_text(render(json.loads(args.results.read_text(encoding="utf-8"))), encoding="utf-8")
        print(f"wrote {args.out}")
        return 0

    args.out = args.out.resolve()
    args.manifest = args.manifest.resolve()
    args.data = args.data.resolve()
    args.emotionedge = str(Path(args.emotionedge).resolve())
    path = args.out / "results.json"
    results = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    results["environment"] = environment(args)
    results["settings"] = {"runs": args.runs, "devices": args.devices, "set": args.set}
    for suite in args.suites:
        {"latency": latency_suite, "transfer": transfer_suite, "emotion": emotion_suite, "mt": mt_suite,
         "overhead": overhead_suite}[suite](args, results)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")  # after every suite
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
