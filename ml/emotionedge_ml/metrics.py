"""Quality metrics for the P5 gate (blueprint: WER · COMET · BLEU · emotion F1 · CCC on V·A·D ·
ECS round-trip · latency). WER, CCC and F1 are implemented here without dependencies; BLEU and
COMET use sacrebleu / unbabel-comet when installed.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Optional, Sequence

_PUNCT = re.compile(r"[^\w\s']", re.UNICODE)


def _words(text: str) -> List[str]:
    return _PUNCT.sub(" ", text.lower()).split()


def edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def wer(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    """Corpus word error rate: total edits / total reference words (case and punctuation ignored)."""
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses differ in length")
    edits = sum(edit_distance(_words(r), _words(h)) for r, h in zip(references, hypotheses))
    words = sum(len(_words(r)) for r in references)
    return edits / words if words else 0.0


def ccc(truth: Sequence[float], pred: Sequence[float]) -> float:
    """Concordance correlation coefficient (Lin 1989), the usual V·A·D regression metric."""
    n = len(truth)
    if n == 0 or n != len(pred):
        raise ValueError("ccc needs two equal-length, non-empty sequences")
    mt = sum(truth) / n
    mp = sum(pred) / n
    vt = sum((t - mt) ** 2 for t in truth) / n
    vp = sum((p - mp) ** 2 for p in pred) / n
    cov = sum((t - mt) * (p - mp) for t, p in zip(truth, pred)) / n
    denom = vt + vp + (mt - mp) ** 2
    return 2 * cov / denom if denom > 0 else 1.0


def macro_f1(truth: Sequence[str], pred: Sequence[str], labels: Optional[Sequence[str]] = None) -> float:
    """Unweighted mean F1 over labels (emotion F1)."""
    if len(truth) != len(pred):
        raise ValueError("truth and pred differ in length")
    labels = list(labels) if labels else sorted(set(truth) | set(pred))
    scores = []
    for label in labels:
        tp = sum(1 for t, p in zip(truth, pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(truth, pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(truth, pred) if t == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def bleu(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    """Corpus BLEU via sacrebleu when available, else a plain 4-gram BLEU with brevity penalty."""
    try:
        import sacrebleu  # type: ignore

        return float(sacrebleu.corpus_bleu(list(hypotheses), [list(references)]).score)
    except ImportError:
        pass
    matches = [0] * 4
    totals = [0] * 4
    ref_len = hyp_len = 0
    for ref, hyp in zip(references, hypotheses):
        r, h = ref.split(), hyp.split()
        ref_len += len(r)
        hyp_len += len(h)
        for n in range(1, 5):
            hc = Counter(tuple(h[i:i + n]) for i in range(len(h) - n + 1))
            rc = Counter(tuple(r[i:i + n]) for i in range(len(r) - n + 1))
            matches[n - 1] += sum(min(c, rc[g]) for g, c in hc.items())
            totals[n - 1] += max(len(h) - n + 1, 0)
    if min(totals) == 0 or min(matches) == 0:
        return 0.0
    log_precision = sum(math.log(m / t) for m, t in zip(matches, totals)) / 4
    brevity = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / max(hyp_len, 1))
    return 100.0 * brevity * math.exp(log_precision)


def comet(sources: Sequence[str], references: Sequence[str], hypotheses: Sequence[str],
          model: str = "Unbabel/wmt22-comet-da") -> Optional[float]:
    """COMET system score with unbabel-comet, or None when it is not installed."""
    try:
        from comet import download_model, load_from_checkpoint  # type: ignore
    except ImportError:
        return None
    scorer = load_from_checkpoint(download_model(model))
    data = [{"src": s, "mt": h, "ref": r} for s, r, h in zip(sources, references, hypotheses)]
    return float(scorer.predict(data, batch_size=16, gpus=0).system_score)


def summarize_vad(truth: Sequence[Sequence[float]], pred: Sequence[Sequence[float]]) -> Dict[str, float]:
    """CCC per axis for lists of (v, a, d) triples."""
    axes = ("valence", "arousal", "dominance")
    return {f"ccc_{name}": ccc([t[i] for t in truth], [p[i] for p in pred]) for i, name in enumerate(axes)}
