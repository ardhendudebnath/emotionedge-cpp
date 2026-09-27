"""Tests for the dependency-free parts of the ML factory: python -m unittest discover ml/tests"""
from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

ML = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML))
sys.path.insert(0, str(ML / "eval"))
sys.path.insert(0, str(ML / "export"))
sys.path.insert(0, str(ML / "train"))

from emotionedge_ml import emotion_space as es  # noqa: E402
from emotionedge_ml import metrics  # noqa: E402
import quality_gate  # noqa: E402
import write_manifest  # noqa: E402


class EmotionSpaceTest(unittest.TestCase):
    def test_ecs_matches_the_blueprint_formula(self):
        src = (-0.62, 0.78, 0.55)
        self.assertAlmostEqual(es.emotion_consistency(src, src), 1.0)
        self.assertAlmostEqual(es.emotion_consistency((-1, -1, -1), (1, 1, 1)), 0.0)
        out = (-0.53, 0.68, 0.55)
        expected = 1 - math.hypot(0.09, 0.10) / (2 * math.sqrt(3))
        self.assertAlmostEqual(es.emotion_consistency(src, out), expected, places=6)

    def test_prototypes_match_the_cpp_core(self):
        # core/emotion/emotion_types.hpp prototype(): the C++ test checks the same file.
        self.assertEqual(es.PROTOTYPES["anger"], (-0.62, 0.78, 0.55))
        self.assertEqual(es.PROTOTYPES["fear"], (-0.45, 0.50, -0.55))
        self.assertEqual(len(es.PROTOTYPES), 7)
        self.assertEqual(es.ECS_TARGET, 0.75)

    def test_dataset_labels_map_onto_the_blueprint_space(self):
        self.assertEqual(es.nearest_label(es.label_to_vad("ANG")), "anger")
        self.assertEqual(es.nearest_label(es.label_to_vad("happy")), "joy")
        self.assertEqual(es.nearest_label((0.05, -0.02, 0.0)), "neutral")
        with self.assertRaises(KeyError):
            es.label_to_vad("bored")


class MetricsTest(unittest.TestCase):
    def test_wer_ignores_case_and_punctuation(self):
        self.assertAlmostEqual(metrics.wer(["I can't believe you did this!"], ["i cant believe you did it"]), 2 / 6)
        self.assertEqual(metrics.wer(["same words"], ["Same, words."]), 0.0)

    def test_ccc(self):
        self.assertAlmostEqual(metrics.ccc([0.1, 0.5, 0.9], [0.1, 0.5, 0.9]), 1.0)
        self.assertLess(metrics.ccc([0.1, 0.5, 0.9], [0.9, 0.5, 0.1]), 0.0)
        shifted = metrics.ccc([0.1, 0.5, 0.9], [0.4, 0.8, 1.2])  # correlated but biased
        self.assertLess(shifted, 0.9)

    def test_macro_f1(self):
        self.assertAlmostEqual(metrics.macro_f1(["anger", "joy"], ["anger", "joy"]), 1.0)
        self.assertAlmostEqual(metrics.macro_f1(["anger", "joy"], ["anger", "anger"]), (2 / 3 + 0) / 2)

    def test_bleu_fallback_is_bounded(self):
        score = metrics.bleu(["the cat sat on the mat"], ["the cat sat on the mat"])
        self.assertGreater(score, 99.0)
        self.assertEqual(metrics.bleu(["a b c d"], ["x y z w"]), 0.0)


class QualityGateTest(unittest.TestCase):
    def status(self, rows, name):
        return next(r for r in rows if r["metric"] == name)["status"]

    def test_detects_regressions_and_target_misses(self):
        baseline = {"ecs": 0.81, "end_to_end_p95_ms": 200.0, "wer": 0.10}
        ok = quality_gate.evaluate({"ecs": 0.805, "end_to_end_p95_ms": 215.0, "wer": 0.102}, baseline)
        self.assertEqual({self.status(ok, m) for m in ("ecs", "end_to_end_p95_ms", "wer")}, {"ok"})
        bad = quality_gate.evaluate({"ecs": 0.78, "end_to_end_p95_ms": 240.0, "wer": 0.2}, baseline)
        self.assertEqual({self.status(bad, m) for m in ("ecs", "end_to_end_p95_ms", "wer")}, {"FAIL"})
        target = quality_gate.evaluate({"ecs": 0.70, "end_to_end_p95_ms": 850.0}, {})
        self.assertEqual(self.status(target, "ecs"), "FAIL")
        self.assertEqual(self.status(target, "end_to_end_p95_ms"), "FAIL")
        self.assertEqual(self.status(target, "comet"), "skipped")

    def test_reads_the_cpp_bench_report(self):
        bench = {"end_to_end_ms": {"p95": 186.4, "count": 9}, "asr_rtf": {"p95": 0.12},
                 "ecs": {"mean": 0.81, "count": 14}, "peak_rss_mb": 53.0, "dropout_samples": 0}
        m = quality_gate.from_bench(bench)
        self.assertEqual(m["end_to_end_p95_ms"], 186.4)
        self.assertEqual(m["ecs"], 0.81)
        self.assertEqual(m["dropout_samples"], 0.0)

    def test_cli_exit_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            baseline = Path(tmp) / "baseline.json"
            baseline.write_text(json.dumps({"metrics": {"ecs": 0.81}}))
            good = Path(tmp) / "good.json"
            good.write_text(json.dumps({"ecs": 0.82}))
            bad = Path(tmp) / "bad.json"
            bad.write_text(json.dumps({"ecs": 0.60}))
            self.assertEqual(quality_gate.main(["--candidate", str(good), "--baseline", str(baseline)]), 0)
            self.assertEqual(quality_gate.main(["--candidate", str(bad), "--baseline", str(baseline)]), 1)


class ManifestTest(unittest.TestCase):
    def test_add_and_rehash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "asr").mkdir()
            (root / "asr" / "model.bin").write_bytes(b"abc")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"schema": 1, "models": []}))
            write_manifest.main(["--manifest", str(manifest), "add", "--id", "asr.test", "--task", "asr",
                                 "--format", "ggml", "--path", "asr/model.bin"])
            entry = json.loads(manifest.read_text())["models"][0]
            self.assertEqual(entry["sha256"], "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
            self.assertEqual(entry["bytes"], 3)
            (root / "asr" / "model.bin").write_bytes(b"abcd")
            write_manifest.main(["--manifest", str(manifest), "rehash"])
            self.assertEqual(json.loads(manifest.read_text())["models"][0]["bytes"], 4)

    def test_repository_manifest_is_consistent(self):
        manifest = json.loads((ML.parent / "models" / "manifest.json").read_text(encoding="utf-8"))
        ids = {m["id"] for m in manifest["models"]}
        for pack in manifest["language_packs"].values():
            self.assertTrue(set(pack.values()) <= ids)
        for m in manifest["models"]:
            self.assertIn(m["task"], {"vad", "asr", "emotion.acoustic", "emotion.lexical", "speaker", "mt", "tts"})
            if m.get("sha256"):
                self.assertRegex(m["sha256"], r"^[0-9a-f]{64}$")


class ControlPrefixTest(unittest.TestCase):
    def test_training_prefix_matches_the_runtime_format(self):
        sys.modules.setdefault("torch", None)  # the module imports torch lazily; the helper does not need it
        import finetune_nllb_lora

        self.assertEqual(finetune_nllb_lora.control_prefix("anger", 0.78, "casual"), "<emo=anger a=0.8 reg=casual>")
        self.assertEqual(finetune_nllb_lora.control_prefix("neutral", -0.04, "formal"), "<emo=neutral a=0.0 reg=formal>")


if __name__ == "__main__":
    unittest.main()
