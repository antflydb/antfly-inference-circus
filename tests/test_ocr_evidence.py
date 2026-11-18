from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ocr_bench as O
import quality_ocr as Q


class OcrEvidenceTests(unittest.TestCase):
    def test_raw_errors_are_not_hidden_by_formatting_diagnostics(self) -> None:
        scores = Q.score_pages(["# TOTAL **84.50**"], ["TOTAL 84.50"], ["invoice"])
        raw = scores["scores"]["raw"]
        self.assertEqual(raw["word_edits"], 2)
        self.assertEqual(raw["reference_words"], 2)
        self.assertEqual(raw["reference_characters"], 11)
        self.assertGreater(raw["character_edits"], 0)
        self.assertEqual(scores["scores"]["markup_normalized"]["wer"], 0)
        altered = Q.score_pages(["# TOTAL **48.50**"], ["TOTAL 84.50"], ["invoice"])
        self.assertEqual(altered["scores"]["text_content_normalized"]["word_edits"], 1)

    def test_case_punctuation_and_layout_have_distinct_diagnostics(self) -> None:
        result = Q.score_pages(["<p>Amount: 42</p>"], ["amount 42"], ["form"])
        self.assertGreater(result["scores"]["raw"]["wer"], 0)
        self.assertGreater(result["scores"]["markup_normalized"]["wer"], 0)
        self.assertEqual(result["scores"]["text_content_normalized"]["wer"], 0)
        self.assertEqual(result["items"][0]["formatting"]["html_tags"], 2)

    def test_corpus_rates_are_micro_averages_with_complete_denominators(self) -> None:
        result = Q.score_pages(["wrong", "two three four"], ["one", "two three four"], ["a", "b"])
        self.assertEqual(result["scores"]["raw"]["wer"], 0.25)
        self.assertEqual(result["scores"]["raw"]["reference_words"], 4)
        self.assertEqual(result["scores"]["raw"]["reference_characters"], 17)
        with self.assertRaises(ValueError):
            Q.score_pages(["a"], ["a", "b"], ["a", "b"])
        with self.assertRaises(ValueError):
            Q.score_pages(["a"], [""], ["a"])

    def test_reference_preserves_annotated_rows_across_semantic_groups(self) -> None:
        def word(text, box, row):
            left, top, right, bottom = box
            return {"text": text, "row_id": row,
                    "quad": {"x1": left, "x2": right, "x3": right, "x4": left,
                             "y1": top, "y2": top, "y3": bottom, "y4": bottom}}
        annotation = {"valid_line": [
            {"words": [word("second", [80, 25, 120, 35], 10), word("lower", [10, 40, 50, 50], 20)]},
            {"words": [word("first", [10, 10, 50, 20], 10), word("", [0, 0, 1, 1], 30)]},
        ]}
        # Real row IDs, not box overlap or semantic entity order, define lines.
        self.assertEqual(Q.reference_text(annotation), "first second\nlower")
        with self.assertRaises(ValueError):
            Q.reference_text({"valid_line": [{"words": [word("bad", [1, 1, 1, 1], 0)]}]})

    def test_paired_quality_differences_preserve_sign_and_denominator(self) -> None:
        providers = {
            "antfly": {"status": "complete", "quality": Q.score_pages(["one wrong"], ["one two"], ["a"])},
            "transformers": {"status": "complete", "quality": Q.score_pages(["one two"], ["one two"], ["a"])},
        }
        raw = Q.comparisons(providers)[0]
        self.assertEqual(raw["wer_difference"], 0.5)
        self.assertEqual(raw["reference_words"], 2)
        self.assertEqual(raw["paired_pages"][0]["wer_difference"], 0.5)
        providers["antfly"]["status"] = "failed"
        self.assertEqual(Q.comparisons(providers), [])

    def test_cache_proof_requires_populated_cache_and_incremental_decode(self) -> None:
        def state(length, populated):
            return {"sequence_length": length, "populated_tensors": populated}
        steps = [{"input_tokens": 8, "use_cache": True, "before": state(0, 0), "after": state(8, 48)},
                 {"input_tokens": 1, "use_cache": True, "before": state(8, 48), "after": state(9, 48)}]
        self.assertTrue(O.validate_cache_probe(steps)["validated"])
        for path, value in [(('input_tokens',), 9), (('before', 'populated_tensors'), 0),
                            (('after', 'sequence_length'), 8), (('use_cache',), False)]:
            broken = copy.deepcopy(steps)
            target = broken[1]
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            with self.subTest(path=path), self.assertRaises(RuntimeError):
                O.validate_cache_probe(broken)

    def test_repetition_agreement_never_reuses_first_output_for_later_call(self) -> None:
        def row(repetition, text):
            return {"call": {"cell": "single_page", "repetition": repetition}, "texts": [text]}
        results = {"antfly": {"rows": [row(0, "alpha"), row(1, "beta")]},
                   "transformers": {"rows": [row(0, "alpha"), row(1, "gamma")]}}
        rows = O.agreement_rows(0, [{"id": "single_page"}], results)
        self.assertTrue(rows[0]["all_normalized_exact"])
        self.assertFalse(rows[1]["all_normalized_exact"])
        results["transformers"]["rows"].pop()
        self.assertFalse(O.agreement_rows(0, [{"id": "single_page"}], results)[1]["complete"])


if __name__ == "__main__":
    unittest.main()
