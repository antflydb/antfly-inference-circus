from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import common  # noqa: E402
import fetch_reference  # noqa: E402
import run_bench  # noqa: E402
import summarize  # noqa: E402


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class CompletionTests(unittest.TestCase):
    def test_missing_or_failed_calls_and_lifecycles_are_not_complete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            write_jsonl(
                output / "lifecycles.jsonl",
                [{"arm": "a", "lc": 0}, {"arm": "b", "lc": 0}],
            )
            complete_calls = [
                {"arm": arm, "lc": 0, "cell": "one", "ok": True}
                for arm in ("a", "b")
                for _ in range(2)
            ]
            write_jsonl(output / "calls.jsonl", complete_calls)
            self.assertTrue(common.benchmark_complete(output, {"a", "b"}, 1, 2))

            write_jsonl(output / "calls.jsonl", complete_calls[:-1])
            self.assertFalse(common.benchmark_complete(output, {"a", "b"}, 1, 2))

            write_jsonl(output / "calls.jsonl", complete_calls[:-1] + [{**complete_calls[-1], "ok": False}])
            self.assertFalse(common.benchmark_complete(output, {"a", "b"}, 1, 2))

            write_jsonl(output / "calls.jsonl", complete_calls)
            write_jsonl(output / "lifecycles.jsonl", [{"arm": "a", "lc": 0}])
            self.assertFalse(common.benchmark_complete(output, {"a", "b"}, 1, 2))

    def test_stale_measurements_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "calls.jsonl").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "prior measurements"):
                common.prepare_output(output)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "speed").mkdir()
            with self.assertRaisesRegex(run_bench.RunnerError, "benchmark records"):
                run_bench.fresh_output(output)

    def test_check_cannot_overwrite_existing_measurement_records(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            retained = output / "result.json"
            original = b'{"status":"complete","measured":true}\n'
            retained.write_bytes(original)
            process = subprocess.run(
                [sys.executable, str(ROOT / "scripts/run_bench.py"), "--check", "--output", str(output)],
                capture_output=True, text=True, check=False,
            )
            self.assertNotEqual(process.returncode, 0)
            self.assertEqual(retained.read_bytes(), original)

    def test_unknown_cell_is_rejected_before_execution(self) -> None:
        with self.assertRaisesRegex(run_bench.RunnerError, "unknown cell"):
            run_bench.validate_selection(["embed"], "not-a-cell", 1)
        with self.assertRaisesRegex(ValueError, "unknown cell"):
            summarize.selected_cells(summarize.WORKLOADS["embed"], "not-a-cell")

    def test_removed_composite_selectors_are_rejected_before_inference(self) -> None:
        removed = (
            ("gliner-base", "long_book_w384", "gliner_bench.py", ["--model", "base"]),
            ("whisper", "batch4", "whisper_bench.py", []),
            ("whisper", "multiwindow60", "whisper_bench.py", []),
        )
        for model, cell, script, model_args in removed:
            with self.subTest(model=model, cell=cell), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / "run"
                process = subprocess.run(
                    [
                        sys.executable, str(ROOT / "scripts/run_bench.py"),
                        "--model", model, "--only-cell", cell, "--output", str(output),
                    ],
                    capture_output=True, text=True, check=False,
                )
                self.assertNotEqual(process.returncode, 0)
                result = json.loads((output / "result.json").read_text())
                direct_output = Path(temporary) / "direct"
                direct = subprocess.run(
                    [
                        sys.executable, str(ROOT / "scripts" / script), "run",
                        *model_args, "--antfly", str(Path(temporary) / "missing-antfly"),
                        "--only-cell", cell, "--out", str(direct_output),
                    ],
                    capture_output=True, text=True, check=False,
                )
                self.assertEqual(direct.returncode, 2)
                self.assertFalse(direct_output.exists())

    def test_short_corpus_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            corpus_path = Path(temporary) / "corpus.json"
            corpus_path.write_text(
                json.dumps(
                    {
                        "sampling": "retained snapshots",
                        "sources": {
                            "darwin": {"sha256": "0" * 64},
                            "wikipedia": {
                                "en": [{"lastrevid": 1, "extract_sha256": "1" * 64}]
                            },
                        },
                        "sentences": [],
                        "paragraphs": [],
                        "pages": [],
                        "chapters": [],
                        "wiki_en": [],
                    }
                ),
                encoding="utf-8",
            )
            with patch.object(common, "CORPUS", corpus_path):
                with self.assertRaisesRegex(ValueError, "minimums"):
                    common.load_corpus()


class SummaryTests(unittest.TestCase):
    SPEC = {
        "workload": "sample",
        "model": "example/model",
        "model_key": "sample",
        "arms": {"a": "provider-a", "b": "provider-b"},
        "cells": {"cell": 2},
    }

    @staticmethod
    def call(arm: str, index: int) -> dict:
        return {
            "arm": arm,
            "lc": 0,
            "cell": "cell",
            "k": index,
            "n_inputs": 1,
            "words": 3,
            "ms": 10.0,
            "cell_wall_s": 0.02,
            "ok": True,
        }

    def test_failed_provider_has_no_speed_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "speed" / "sample"
            write_jsonl(run / "lifecycles.jsonl", [{"arm": "a", "lc": 0}, {"arm": "b", "lc": 0}])
            write_jsonl(
                run / "calls.jsonl",
                [self.call("a", 0), self.call("a", 1), self.call("b", 0)],
            )
            steps, _ = summarize.speed_summary(root, "sample", self.SPEC, {"cell": 2}, 1)
            by_provider = {step["provider"]: step for step in steps}
            self.assertEqual(by_provider["provider-a"]["status"], "complete")
            self.assertIn("metrics", by_provider["provider-a"])
            self.assertEqual(by_provider["provider-b"]["status"], "failed")
            self.assertNotIn("metrics", by_provider["provider-b"])

    def test_invalid_memory_sampler_suppresses_metric(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "memory" / "sample" / "cell"
            lifecycle_rows = [
                {
                    "arm": arm,
                    "lc": 0,
                    "footprint": {
                        "valid": arm == "a",
                        "metric_version": "test-v1",
                        "memory_peak_lower_bytes": 100,
                        "memory_peak_upper_bytes": 100,
                        "peak_rss_bytes": 80,
                        "method": "test",
                    },
                }
                for arm in ("a", "b")
            ]
            write_jsonl(run / "lifecycles.jsonl", lifecycle_rows)
            write_jsonl(
                run / "calls.jsonl",
                [self.call("a", 0), self.call("a", 1), self.call("b", 0), self.call("b", 1)],
            )
            steps = summarize.memory_summary(root, "sample", self.SPEC, {"cell": 2}, lifecycles=1)
            by_provider = {step["provider"]: step for step in steps}
            self.assertEqual(by_provider["provider-a"]["status"], "complete")
            self.assertEqual(by_provider["provider-b"]["status"], "failed")
            self.assertNotIn("metrics", by_provider["provider-b"])


class ReferenceVerificationTests(unittest.TestCase):
    def test_reference_weight_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            models = root / "models"
            model = models / "owner" / "name"
            model.mkdir(parents=True)
            expected_bytes = b"expected"
            expected_hash = hashlib.sha256(expected_bytes).hexdigest()
            (model / "model.safetensors").write_bytes(expected_bytes)
            (model / ".antfly-download-complete.json").write_text(
                json.dumps(
                    {
                        "source": {"owner": "owner", "name": "name", "resolved_revision": "revision"},
                        "artifacts": [{"path": "model.safetensors", "sha256": expected_hash}],
                    }
                ),
                encoding="utf-8",
            )
            reference = root / "reference"
            reference.mkdir()
            (reference / "model.safetensors").write_bytes(b"wrong")
            references = root / "references.json"
            references.write_text(json.dumps([{"repo": "owner/name", "revision": "revision"}]), encoding="utf-8")
            with patch.object(fetch_reference.C, "REFERENCES", references), patch.object(
                fetch_reference.C, "reference_paths", return_value={"owner/name": str(reference)}
            ):
                with self.assertRaisesRegex(RuntimeError, "weight mismatch"):
                    fetch_reference.verify_cached(models, {"owner/name"})


if __name__ == "__main__":
    unittest.main()
