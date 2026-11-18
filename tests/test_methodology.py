from __future__ import annotations

import hashlib
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyze
import common as C
import precision
import summarize as S


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


class EvidenceTests(unittest.TestCase):
    SPEC = {"workload": "sample", "model": "owner/model", "model_key": "sample",
            "arms": {"antfly": "antfly", "py": "reference"}, "cells": {"small": 2, "large": 2}}

    @staticmethod
    def calls(lifecycles=3):
        return [{"arm": arm, "lc": lc, "cell": cell, "k": k, "ok": True, "ms": 10 if arm == "antfly" else 20,
                 "n_inputs": 1, "words": 2, "cell_wall_s": 0.02}
                for arm in ("antfly", "py") for lc in range(lifecycles) for cell in ("small", "large") for k in range(2)]

    def test_refused_cell_preserves_complete_sibling_without_refusal_ratio(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "speed" / "sample"
            calls = self.calls()
            for row in calls:
                if row["arm"] == "antfly" and row["cell"] == "large":
                    row.update(ok=False, status=400, response_error={"code": "BatchTooLarge"})
            jsonl(run / "calls.jsonl", calls)
            jsonl(run / "lifecycles.jsonl", [{"arm": arm, "lc": lc} for arm in ("antfly", "py") for lc in range(3)])
            steps, pairs = S.speed_summary(root, "sample", self.SPEC, self.SPEC["cells"], 3)
            native = next(step for step in steps if step["provider"] == "antfly")
            self.assertEqual(native["status"], "partial")
            self.assertEqual(native["cells"]["large"]["status"], "refused")
            self.assertEqual(native["cells"]["large"]["refused_calls"], 6)
            self.assertEqual(set(native["metrics"]["cells"]), {"small"})
            self.assertEqual([pair["cell"] for pair in pairs], ["small"])
            self.assertEqual(pairs[0]["metrics"]["p50_ratio"]["median"], 2)
            self.assertTrue(C.benchmark_complete(run, {"antfly", "py"}, 3, 4))
            calls[-1].update(ok=False, status=500, response_error={"code": "BatchTooLarge"})
            jsonl(run / "calls.jsonl", calls)
            self.assertFalse(C.benchmark_complete(run, {"antfly", "py"}, 3, 4))

    def test_successful_responses_after_refused_warmup_do_not_become_warm_timings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "speed" / "sample"
            calls = self.calls()
            for row in calls:
                if row["arm"] == "antfly" and row["cell"] == "large":
                    row["warmup_state"] = "refused"
            jsonl(run / "calls.jsonl", calls)
            jsonl(run / "lifecycles.jsonl", [{"arm": arm, "lc": lc} for arm in ("antfly", "py") for lc in range(3)])
            steps, pairs = S.speed_summary(root, "sample", self.SPEC, self.SPEC["cells"], 3)
            native = next(step for step in steps if step["provider"] == "antfly")
            self.assertEqual(native["cells"]["large"]["successful_calls"], 6)
            self.assertEqual(native["cells"]["large"]["status"], "failed")
            self.assertEqual(set(native["metrics"]["cells"]), {"small"})
            self.assertEqual([pair["cell"] for pair in pairs], ["small"])

    def test_incomplete_and_server_failure_are_not_request_refusals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jsonl(root / "lifecycles.jsonl", [{"arm": "antfly", "lc": 0}])
            rows = [row for row in self.calls(1) if row["arm"] == "antfly" and row["cell"] == "large"]
            for row in rows:
                row.update(ok=False, status=500, response_error={"message": "limit"})
            jsonl(root / "calls.jsonl", rows)
            self.assertEqual(S.cell_completion(root, "antfly", "large", 2, 1)["status"], "failed")
            for row in rows:
                row["status"] = 413
            jsonl(root / "calls.jsonl", rows[:-1])
            self.assertEqual(S.cell_completion(root, "antfly", "large", 2, 1)["status"], "failed")
            rows[0]["response_error"] = {}
            self.assertFalse(C.call_refused(rows[0]))

    def test_three_memory_lifecycles_preserve_bounds_and_observed_spread(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for cell in self.SPEC["cells"]:
                run = root / "memory" / "sample" / cell
                jsonl(run / "calls.jsonl", [row for row in self.calls() if row["cell"] == cell])
                rows = []
                for arm, base in (("antfly", 100), ("py", 200)):
                    for lc, offset in enumerate((0, 40, 20)):
                        rows.append({"arm": arm, "lc": lc, "footprint": {
                            "valid": True, "metric_version": "bounded-v1",
                            "memory_peak_lower_bytes": base + offset, "memory_peak_upper_bytes": base + offset + 60}})
                jsonl(run / "lifecycles.jsonl", rows)
            steps = S.memory_summary(root, "sample", self.SPEC, self.SPEC["cells"], 3)
            metric = steps[0]["metrics"]["cells"]["small"]
            self.assertEqual(metric["memory_peak_lower_bytes"], {"median": 120, "min": 100, "max": 140})
            pair = S.memory_comparisons(steps, self.SPEC, self.SPEC["cells"])[0]
            self.assertNotIn("reference_to_antfly_ratio", pair["metrics"])
            interval = pair["metrics"]["reference_to_antfly_ratio_interval"]
            self.assertAlmostEqual(interval["lower"], 220 / 180)
            self.assertAlmostEqual(interval["upper"], 280 / 120)
            run = root / "memory" / "sample" / "large"
            rows = C.load_jsonl(run / "lifecycles.jsonl")
            rows.pop()
            jsonl(run / "lifecycles.jsonl", rows)
            steps = S.memory_summary(root, "sample", self.SPEC, self.SPEC["cells"], 3)
            reference = next(step for step in steps if step["provider"] == "reference")
            self.assertEqual(reference["status"], "partial")
            self.assertEqual(set(reference["metrics"]["cells"]), {"small"})

    def test_structured_agreement_never_overwrites_an_earlier_lifecycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for lc in range(2):
                for arm in ("antfly", "py"):
                    entity = [0, 4, "person"] if lc == 1 or arm == "antfly" else [5, 9, "person"]
                    rows.append({"cell": "small", "lc": lc, "k": 0, "j": 0, "arm": arm,
                                 "input_sha256": "a" * 64, "o": {"e": [entity], "c": {}, "r": []}})
            jsonl(root / "outputs.jsonl", rows)
            metric = analyze.agreement(root)["antfly_vs_py"]["small"]
            self.assertEqual(metric["paired_inputs"], 2)
            self.assertEqual(metric["exact_entity_inputs"], 1)
            self.assertEqual(metric["span_f1"], 0.5)
            rows[-1]["input_sha256"] = "b" * 64
            jsonl(root / "outputs.jsonl", rows)
            metric = analyze.agreement(root)["antfly_vs_py"]["small"]
            self.assertEqual(metric["paired_inputs"], 1)
            self.assertEqual(metric["invalid_or_unpaired_inputs"], 1)

    def test_source_policy_requires_matching_binary_and_exact_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = root / "antfly"
            binary.write_bytes(b"a distinct executable")
            digest = hashlib.sha256(binary.read_bytes()).hexdigest()
            self.assertEqual(precision.executable_digest(binary), digest)
            commit = "a" * 40
            policy = {"parameter_dtype": "mixed", "sources": [f"https://github.com/antflydb/antfly/blob/{commit}/loader.zig"]}
            path = root / "evidence.json"
            path.write_text(json.dumps({"schema_version": 1, "executables": {digest: {"source_commit": commit,
                "models": {"owner/model": {"metal": policy}}}}}))
            result = precision.load_native_evidence(path, digest, "owner/model", "metal")
            self.assertEqual(result["evidence_kind"], "executable_bound_source_policy")
            self.assertIsNone(precision.load_native_evidence(path, "b" * 64, "owner/model", "metal"))
            policy["sources"] = ["https://github.com/antflydb/antfly/blob/main/loader.zig"]
            path.write_text(json.dumps({"schema_version": 1, "executables": {digest: {"source_commit": commit,
                "models": {"owner/model": {"metal": policy}}}}}))
            with self.assertRaises(ValueError):
                precision.load_native_evidence(path, digest, "owner/model", "metal")

    def test_checkpoint_inventory_counts_storage_without_inventing_compute_dtype(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.safetensors"
            header = json.dumps({"weight": {"dtype": "BF16", "shape": [2, 3], "data_offsets": [0, 12]},
                                 "norm": {"dtype": "F32", "shape": [2], "data_offsets": [12, 20]}}).encode()
            path.write_bytes(struct.pack("<Q", len(header)) + header + bytes(20))
            result = precision.safetensors_inventory(path)
            self.assertEqual(result["elements_by_dtype"], {"BF16": 6, "F32": 2})
            self.assertEqual(result["tensor_count_by_dtype"], {"BF16": 1, "F32": 1})
            self.assertNotIn("compute_dtype", result)
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaises(ValueError):
                precision.safetensors_inventory(path)

    def test_practical_tie_is_a_symmetric_five_percent_ratio_band(self):
        for value in (1 / 1.05, 1.0, 1.05):
            self.assertEqual(S.practical_category(value), "practical_tie")
        self.assertEqual(S.practical_category(1.05001), "antfly_faster")
        self.assertEqual(S.practical_category(1 / 1.05001), "reference_faster")


if __name__ == "__main__":
    unittest.main()
