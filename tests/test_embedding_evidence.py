from __future__ import annotations

import gzip
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import embed_bench as E
import gliner_bench as G
import quality_embed as Q


class EmbeddingEvidenceTests(unittest.TestCase):
    def test_all_batch_inputs_and_lifecycles_contribute_to_agreement(self):
        cells = [{"id": "batch", "calls": [[f"call{k}-input{j}" for j in range(6)] for k in range(3)]}]
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            for lifecycle in range(2):
                for arm in ("antfly", "py"):
                    with gzip.open(E.vector_spool(out, arm, lifecycle), "wt", encoding="utf-8") as stream:
                        for call, texts in enumerate(cells[0]["calls"]):
                            values = [[1.0, 0.0] for _ in texts]
                            # Only the last input of the last call of the second lifecycle changes.
                            if lifecycle == 1 and arm == "antfly" and call == 2:
                                values[5] = [0.0, 1.0]
                            E.capture_vectors(stream, arm, lifecycle, "batch", call, texts, values)
            E.write_agreement(out, cells, 2)
            first, second = [json.loads(line) for line in (out / "agreement.jsonl").read_text().splitlines()]
            self.assertEqual(first["paired_vectors"], 18)
            self.assertEqual(second["paired_vectors"], 18)
            self.assertEqual(first["exact_vectors"], 18)
            self.assertEqual(second["exact_vectors"], 17)
            self.assertEqual(second["cosine_below_0_99"], 1)
            self.assertAlmostEqual(second["cosine_mean"], 17 / 18)
            self.assertEqual(second["diagnostics"][0]["k"], 2)
            self.assertEqual(second["diagnostics"][0]["j"], 5)
            self.assertEqual(second["lc"], 1)
            self.assertTrue(second["complete"])
            self.assertFalse(any(out.glob(".vectors-*")))

    def test_missing_invalid_dimension_and_input_mismatch_keep_denominator(self):
        cells = [{"id": "batch", "calls": [["a", "b", "c", "d", "e"]]}]
        records = {"antfly": {}, "py": {}}
        for arm in records:
            stream = io.StringIO()
            E.capture_vectors(stream, arm, 4, "batch", 0, cells[0]["calls"][0], [[1.0, 0.0]] * 5)
            for line in stream.getvalue().splitlines():
                row = json.loads(line)
                records[arm][("batch", row["k"], row["j"])] = row
        del records["antfly"][("batch", 0, 1)]
        records["antfly"][("batch", 0, 2)]["v"] = [float("nan"), 1.0]
        records["antfly"][("batch", 0, 3)]["v"] = [1.0, 0.0, 0.0]
        records["antfly"][("batch", 0, 4)]["input_sha256"] = E.input_sha256("wrong input")
        row = E.paired_agreement(cells, 4, records)[0]
        self.assertEqual(row["expected_vectors"], 5)
        self.assertEqual(row["paired_vectors"], 1)
        self.assertEqual(row["missing_antfly"], 1)
        self.assertEqual(row["invalid_antfly"], 1)
        self.assertEqual(row["dimension_mismatches"], 1)
        self.assertEqual(row["input_mismatches"], 1)
        self.assertFalse(row["complete"])

    def test_failed_arm_still_reports_all_expected_vectors(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            E.write_agreement(out, [{"id": "single", "calls": [["a"], ["b", "c"]]}], 1)
            row = json.loads((out / "agreement.jsonl").read_text())
            self.assertEqual(row["expected_vectors"], 3)
            self.assertEqual(row["missing_antfly"], 3)
            self.assertEqual(row["missing_py"], 3)
            self.assertIsNone(row["cosine_mean"])
            self.assertFalse(row["complete"])

    def test_capture_preserves_full_vector_precision(self):
        stream = io.StringIO()
        value = 0.12345678912345678
        E.capture_vectors(stream, "py", 2, "single", 7, ["café"], [[value, 1.0]])
        row = json.loads(stream.getvalue())
        self.assertEqual(row["v"][0], value)
        self.assertEqual(row["input_sha256"], E.input_sha256("café"))
        self.assertEqual((row["arm"], row["lc"], row["k"], row["j"]), ("py", 2, 7, 0))


class RetrievalEvidenceTests(unittest.TestCase):
    def fixture(self):
        return {"documents": [{"id": key} for key in ("a", "b", "c")],
                "queries": [{"id": "q1", "relevant": {"a": 1, "b": 1}},
                            {"id": "q2", "relevant": {"c": 1}}]}

    def test_absolute_scores_use_judgments_macro_recall_and_full_rank_mrr(self):
        score = Q.score_vectors(self.fixture(), [[1, 0], [0, 1], [-1, 0]], [[1, 0], [0, 1]])
        metrics = score["metrics"]
        self.assertEqual(metrics["relevant_documents"], 3)
        self.assertEqual(metrics["hits@1"], 1)
        self.assertEqual(metrics["recall@1"], 0.25)  # macro, not one pooled hit / three labels
        self.assertEqual(metrics["recall@5"], 1)
        self.assertAlmostEqual(metrics["mrr"], (1 + 1 / 3) / 2)
        self.assertEqual(score["queries"][1]["first_relevant_rank"], 3)
        self.assertEqual(score["queries"][1]["top10"], ["b", "a", "c"])

    def test_paired_scores_report_signed_task_quality_difference(self):
        fixture = self.fixture()
        native = Q.score_vectors(fixture, [[1, 0], [0, 1], [-1, 0]], [[1, 0], [-1, 0]])
        reference = Q.score_vectors(fixture, [[1, 0], [0, 1], [-1, 0]], [[1, 0], [0, 1]])
        comparison = Q.compare_scores(native, reference)
        self.assertEqual(comparison["paired_queries"], 2)
        self.assertAlmostEqual(comparison["differences"]["mrr"], 1 / 3)
        self.assertEqual(comparison["differences"]["recall@1"], 0.5)
        self.assertAlmostEqual(comparison["queries"][1]["reciprocal_rank"], 2 / 3)
        reference["queries"].pop()
        with self.assertRaisesRegex(ValueError, "identical query sets"):
            Q.compare_scores(native, reference)

    def test_cosine_is_scale_invariant_and_ties_break_by_document_id(self):
        fixture = {"documents": [{"id": "z"}, {"id": "a"}],
                   "queries": [{"id": "q", "relevant": {"a": 1}}]}
        scores = Q.score_vectors(fixture, [[1, 0], [10, 0]], [[3, 0]])
        self.assertEqual(scores["queries"][0]["top10"], ["a", "z"])
        self.assertEqual(scores["metrics"]["mrr"], 1)

    def test_invalid_vectors_and_missing_judged_documents_fail(self):
        fixture = {"documents": [{"id": "a"}], "queries": [{"id": "q", "relevant": {"a": 1}}]}
        for documents, queries in (([[0, 0]], [[1, 0]]), ([[1, 0]], [[1]]),
                                   ([[1, 0]], [[float("inf"), 0]]), ([], [[1, 0]])):
            with self.assertRaises(ValueError):
                Q.score_vectors(fixture, documents, queries)
        fixture["queries"][0]["relevant"] = {"missing": 1}
        with self.assertRaisesRegex(ValueError, "missing relevance judgments"):
            Q.score_vectors(fixture, [[1, 0]], [[1, 0]])

    def test_offline_check_rejects_tampered_public_source_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "corpus.jsonl.gz").write_bytes(b"changed public corpus")
            with self.assertRaisesRegex(ValueError, "public dataset hash mismatch: corpus.jsonl.gz"):
                Q.check(root)


class ExtractionEvidenceTests(unittest.TestCase):
    def test_unicode_spans_classifications_and_relations_remain_distinct(self):
        text = "Zoë in Paris"
        native = {"entities": [{"label": "person", "start": 0, "end": 4, "score": 0.9}],
                  "classifications": [{"name": "topic", "label": "travel"},
                                      {"name": "tags", "label": "city"}, {"name": "tags", "label": "person"}],
                  "relations": [{"type": "located_in", "source": {"start": 0, "end": 4},
                                 "target": {"start": 8, "end": 13}}]}
        normalized = G.normalize_antfly(native, text)
        self.assertEqual(normalized["e"], [["person", 0, 3, 0.9]])
        self.assertEqual(normalized["c"], {"topic": "travel", "tags": ["city", "person"]})
        self.assertEqual(normalized["r"], [["located_in", 0, 3, 7, 12]])
        upstream = {"entities": {"person": [{"start": 0, "end": 3, "confidence": 0.9}]},
                    "topic": "travel", "tags": ["city", "person"],
                    "relation_extraction": {"located_in": [{"head": {"start": 0, "end": 3},
                                                              "tail": {"start": 7, "end": 12}}]}}
        self.assertEqual(G.normalize_python(upstream, {"classifications": [{"name": "topic"}, {"name": "tags"}]}), normalized)

    def test_refused_decide_batch_is_not_counted_as_successful_empty_outputs(self):
        texts = [f"hotel request {index}" for index in range(32)]
        cell = {"id": "decide_b32", "schema": G.DECIDE_SCHEMA}
        rows = list(G.output_rows("antfly", "decide", 2, cell, 3, texts, None, 413, "batch size refused"))
        self.assertEqual({(row["lc"], row["k"], row["j"]) for row in rows},
                         {(2, 3, index) for index in range(32)})
        for row, text in zip(rows, texts):
            self.assertIsNone(row["o"])
            self.assertEqual(row["status"], 413)
            self.assertEqual(row["err"], "batch size refused")
            self.assertEqual(row["input_sha256"], E.input_sha256(text))


if __name__ == "__main__":
    unittest.main()
