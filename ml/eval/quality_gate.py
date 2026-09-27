#!/usr/bin/env python3
"""P5 quality gate: "every phase ends with the P5 quality gate: WER, COMET, emotion F1, ECS and
latency must not regress" (blueprint p.4).

    python ml/eval/quality_gate.py --candidate metrics.json [--bench out/bench/bench.json]
                                   [--baseline ml/eval/baseline.json] [--update-baseline]

`metrics.json` holds offline quality metrics (see emotionedge_ml.metrics); `--bench` reads the
JSON written by the C++ latency harness (ee_bench). A metric fails if it regresses past its
tolerance against the baseline, or misses the blueprint's absolute success targets. Metrics
absent from the candidate are reported as skipped. Exit status 1 on any failure.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Rule:
    higher_is_better: bool
    tolerance: float          # allowed regression
    relative: bool = False    # tolerance is a fraction of the baseline
    target: Optional[float] = None  # blueprint success target (absolute)


RULES: Dict[str, Rule] = {
    "wer": Rule(False, 0.005),
    "bleu": Rule(True, 0.5),
    "comet": Rule(True, 0.005),
    "emotion_f1": Rule(True, 0.01),
    # ml/eval/eval_emotion.py on the C++ models: RAVDESS (voice) and MELD test (words).
    "emotion_uar_acoustic": Rule(True, 0.01),
    "emotion_uar_lexical": Rule(True, 0.01),
    "chrf": Rule(True, 0.5),
    "ccc_valence": Rule(True, 0.01),
    "ccc_arousal": Rule(True, 0.01),
    "ccc_dominance": Rule(True, 0.01),
    "ecs": Rule(True, 0.01, target=0.75),
    "end_to_end_p95_ms": Rule(False, 0.10, relative=True, target=800.0),
    "asr_rtf_p95": Rule(False, 0.10, relative=True, target=0.3),
    "peak_rss_mb": Rule(False, 0.10, relative=True, target=3072.0),
    "dropout_samples": Rule(False, 0.0, target=0.0),
}


def from_bench(bench: dict) -> Dict[str, float]:
    """Metrics from the ee_bench JSON report."""
    out: Dict[str, float] = {}
    if bench.get("end_to_end_ms", {}).get("count"):
        out["end_to_end_p95_ms"] = float(bench["end_to_end_ms"]["p95"])
    if "asr_rtf" in bench:
        out["asr_rtf_p95"] = float(bench["asr_rtf"]["p95"])
    if bench.get("ecs", {}).get("count"):
        out["ecs"] = float(bench["ecs"]["mean"])
    if "peak_rss_mb" in bench:
        out["peak_rss_mb"] = float(bench["peak_rss_mb"])
    if "dropout_samples" in bench:
        out["dropout_samples"] = float(bench["dropout_samples"])
    return out


def evaluate(candidate: Dict[str, float], baseline: Dict[str, float]) -> List[dict]:
    rows = []
    for name, rule in RULES.items():
        if name not in candidate:
            rows.append({"metric": name, "status": "skipped"})
            continue
        value = float(candidate[name])
        problems = []
        if name in baseline:
            base = float(baseline[name])
            slack = rule.tolerance * abs(base) if rule.relative else rule.tolerance
            regressed = value < base - slack if rule.higher_is_better else value > base + slack
            if regressed:
                problems.append(f"regressed from {base:g}")
        if rule.target is not None:
            if rule.higher_is_better and value < rule.target:
                problems.append(f"below target {rule.target:g}")
            if not rule.higher_is_better:
                missed = value > rule.target if rule.target == 0.0 else value >= rule.target
                if missed:
                    problems.append(f"misses target {'0' if rule.target == 0.0 else '< ' + format(rule.target, 'g')}")
        rows.append({"metric": name, "value": value, "baseline": baseline.get(name),
                     "status": "FAIL" if problems else "ok", "detail": "; ".join(problems)})
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate", type=Path, help="offline quality metrics JSON")
    parser.add_argument("--bench", type=Path, help="ee_bench JSON report")
    parser.add_argument("--baseline", type=Path, default=HERE / "baseline.json")
    parser.add_argument("--update-baseline", action="store_true", help="write the candidate as the new baseline")
    args = parser.parse_args(argv)

    candidate: Dict[str, float] = {}
    if args.candidate:
        candidate.update({k: float(v) for k, v in json.loads(args.candidate.read_text()).items() if isinstance(v, (int, float))})
    if args.bench:
        candidate.update(from_bench(json.loads(args.bench.read_text())))
    if not candidate:
        parser.error("give --candidate and/or --bench")

    baseline_doc = json.loads(args.baseline.read_text()) if args.baseline.exists() else {}
    baseline = {k: v for k, v in baseline_doc.get("metrics", {}).items()}
    rows = evaluate(candidate, baseline)

    print(f"{'metric':<20} {'value':>12} {'baseline':>12}  status")
    for r in rows:
        if r["status"] == "skipped":
            print(f"{r['metric']:<20} {'-':>12} {'-':>12}  skipped")
            continue
        base = "-" if r["baseline"] is None else f"{r['baseline']:.4g}"
        print(f"{r['metric']:<20} {r['value']:>12.4g} {base:>12}  {r['status']}  {r['detail']}")
    failed = [r for r in rows if r["status"] == "FAIL"]

    if args.update_baseline:
        baseline_doc["metrics"] = {**baseline, **candidate}
        args.baseline.write_text(json.dumps(baseline_doc, indent=2) + "\n")
        print(f"baseline updated: {args.baseline}")
    print("quality gate: " + ("FAILED" if failed else "passed"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
