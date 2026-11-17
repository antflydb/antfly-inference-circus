from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import quality_ocr as O
import quality_whisper as W


class QualityFailureTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"call": {"cell": "a", "n_inputs": 1, "ok": True, "status": 200}, "texts": ["one"], "refs": ["one two"]},
            {"call": {"cell": "b", "n_inputs": 1, "ok": True, "status": 200}, "texts": ["three"], "refs": ["three"]},
        ]
        self.cells = [{"id": row["call"]["cell"], "calls": [{"id": row["call"]["cell"], "ref": row["refs"][0]}]}
                      for row in self.rows]
        self.manifest = {"audio": [{"id": row["call"]["cell"], "text": row["refs"][0]} for row in self.rows]}

    def outcomes(self, info):
        return ((O.provider_outcome(info, self.cells), "pages"), (W.provider_outcome(info, self.manifest), "clips"))

    @staticmethod
    def refuse(row):
        row["call"].update(ok=False, status=413, response_error={"code": "BUDGET_EXCEEDED"})
        row["texts"] = []
        row["refs"] = []

    def test_mixed_success_and_refusal_retains_denominators_without_partial_accuracy(self):
        rows = copy.deepcopy(self.rows)
        self.refuse(rows[1])
        for outcome, unit in self.outcomes({"rows": rows}):
            with self.subTest(unit=unit):
                self.assertEqual(outcome["status"], "failed")
                self.assertEqual(outcome["coverage"][f"expected_{unit}"], 2)
                self.assertEqual(outcome["coverage"][f"successful_{unit}"], 1)
                self.assertEqual(outcome["coverage"][f"refused_{unit}"], 1)
                self.assertNotIn("scores", outcome)
                self.assertNotIn("quality", outcome)
                self.assertEqual(outcome["requests"][1]["status"], 413)

    def test_complete_actual_refusals_are_distinct_from_missing_operations_and_cleanup_failure(self):
        rows = copy.deepcopy(self.rows)
        for row in rows:
            self.refuse(row)
        for outcome, unit in self.outcomes({"rows": rows}):
            self.assertEqual(outcome["status"], "refused")
            self.assertEqual(outcome["coverage"][f"refused_{unit}"], 2)
        for outcome, unit in self.outcomes({"rows": rows[:1]}):
            self.assertEqual(outcome["status"], "failed")
            self.assertEqual(outcome["coverage"][f"missing_{unit}"], 1)
        for outcome, _ in self.outcomes({"rows": rows, "error": "owned process cleanup failed"}):
            self.assertEqual(outcome["status"], "failed")

    def test_cleanup_failure_preserves_full_scored_outputs_without_claiming_complete_lifecycle(self):
        info = {"rows": copy.deepcopy(self.rows), "error": "owned process cleanup failed"}
        before = copy.deepcopy(info)
        for outcome, unit in self.outcomes(info):
            self.assertEqual(outcome["status"], "failed")
            score = outcome["quality"]["scores"]["raw"] if unit == "pages" else outcome["scores"]
            self.assertEqual(score["word_edits"], 1)
            self.assertEqual(score["reference_words"], 3)
            self.assertAlmostEqual(score["wer"], 1 / 3)
            self.assertIn("cleanup failed", outcome["error"])
            self.assertNotIn("one two", json.dumps(outcome))
        self.assertEqual(info, before)


if __name__ == "__main__":
    unittest.main()
