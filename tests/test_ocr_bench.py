from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("ocr_bench_test_module", ROOT / "scripts" / "ocr_bench.py")
ocr = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ocr)


class OcrBenchTests(unittest.TestCase):
    def test_quality_normalizes_markdown_but_counts_real_errors(self) -> None:
        exact = ocr.quality(["# TERMITE SUPPLY COMPANY\nTOTAL $84.50"],
                            ["TERMITE SUPPLY COMPANY\nTOTAL 84.50"])
        self.assertEqual(exact["wer"], 0)
        self.assertEqual(exact["cer"], 0)
        self.assertEqual(exact["exact_normalized_count"], 1)
        changed = ocr.quality(["TERMITE SUPPLY COMPANY TOTAL 48.50"],
                              ["TERMITE SUPPLY COMPANY TOTAL 84.50"])
        self.assertGreater(changed["wer"], 0)
        self.assertGreater(changed["cer"], 0)

    def test_rendered_cells_are_matched_single_and_batch_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cells = ocr.make_cells(None, root)
            self.assertEqual([cell["id"] for cell in cells], ocr.CELLS)
            self.assertEqual(len(cells[0]["calls"]), 1)
            self.assertEqual(len(cells[1]["calls"]), 4)
            self.assertEqual(cells[0]["calls"][0]["sha256"], cells[1]["calls"][0]["sha256"])
            self.assertTrue(all(call["path"].is_file() for call in cells[1]["calls"]))
            self.assertEqual([call["ref"] for call in cells[1]["calls"]], ocr.PAGE_TEXTS)
            for call in cells[1]["calls"]:
                left, top, right, bottom = call["text_bbox"]
                self.assertGreaterEqual(left, 90)
                self.assertGreaterEqual(top, 100)
                self.assertLessEqual(right, 934)
                self.assertLessEqual(bottom, 1240)
                self.assertEqual(ocr.normalize_words(call["rendered_text"]),
                                 ocr.normalize_words(call["ref"]))
            self.assertTrue(any(call["rendered_lines"] > len(call["ref"].splitlines())
                                for call in cells[1]["calls"]))


    def test_agreement_reports_exact_and_character_similarity(self) -> None:
        cell = {"id": "single_page", "calls": [{"ref": "alpha beta"}]}
        equal = {
            "antfly": {"rows": [{"call": {"cell": "single_page"}, "texts": ["# ALPHA BETA"]}]},
            "transformers": {"rows": [{"call": {"cell": "single_page"}, "texts": ["alpha beta"]}]},
        }
        row = ocr.agreement_rows(1, [cell], equal)[0]
        self.assertTrue(row["complete"])
        self.assertTrue(row["all_normalized_exact"])
        self.assertEqual(row["character_agreement"], 1)
        unequal = dict(equal)
        unequal["transformers"] = {"rows": [{"call": {"cell": "single_page"}, "texts": ["alpha zeta"]}]}
        row = ocr.agreement_rows(1, [cell], unequal)[0]
        self.assertFalse(row["all_normalized_exact"])
        self.assertLess(row["character_agreement"], 1)


if __name__ == "__main__":
    unittest.main()
