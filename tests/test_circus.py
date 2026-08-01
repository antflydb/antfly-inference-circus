from __future__ import annotations

import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("circus", ROOT / "scripts" / "circus.py")
circus = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(circus)


class MatrixTests(unittest.TestCase):
    def setUp(self) -> None:
        self.matrix = circus.read_json(ROOT / "matrix.json")
        self.fixtures = circus.read_json(ROOT / "fixtures" / "cases.json")

    def test_matrix_is_valid_and_all_working_categories_have_smoke(self) -> None:
        circus.validate_matrix(self.matrix)
        self.assertEqual(self.matrix["antfly_contract"], "v0.2.0-rc.24")
        selected = circus.select_benchmarks(self.matrix, "smoke", None)
        self.assertEqual(set(circus.CATEGORY_ORDER), {bench["category"] for bench in selected})
        self.assertTrue(all(circus.providers_for(bench, "smoke") == ["antfly"] for bench in selected))

    def test_full_has_local_reference_providers(self) -> None:
        selected = circus.select_benchmarks(self.matrix, "full", None)
        providers = {provider for bench in selected for provider in circus.providers_for(bench, "full")}
        self.assertIn("llama.cpp", providers)
        self.assertIn("ollama", providers)
        self.assertIn("vllm-metal", providers)
        self.assertIn("gliner2", providers)
        self.assertIn("faster-whisper", providers)

    def test_rewriter_is_not_benchmarked(self) -> None:
        self.assertNotIn("rewriting", {bench["category"] for bench in self.matrix["benchmarks"]})

    def test_quality_gates(self) -> None:
        ok, _, margin, unit = circus.quality_result(
            "dense embedding",
            {"data": [
                {"index": 0, "embedding": [1.0, 0.0]},
                {"index": 1, "embedding": [0.9, 0.1]},
                {"index": 2, "embedding": [0.0, 1.0]},
            ]},
            self.fixtures["embedding"],
        )
        self.assertTrue(ok)
        self.assertGreater(margin, 0)
        self.assertEqual(unit, "cosine")
        ok, _, _, _ = circus.quality_result(
            "reranking",
            {"data": [{"index": 0, "score": 0.9}, {"index": 1, "score": 0.1}, {"index": 2, "score": 0.5}]},
            self.fixtures["reranking"],
        )
        self.assertTrue(ok)

    def test_rc24_request_shapes(self) -> None:
        extraction = circus.select_benchmarks(self.matrix, "smoke", "extraction")[0]
        path, body = circus.antfly_request(extraction, self.fixtures["extraction"])
        self.assertEqual(path, "/ai/v1/extract")
        self.assertEqual(body["inputs"], [{"content": self.fixtures["extraction"]["texts"][0]}])
        self.assertIn("structures", body["schema"])
        self.assertEqual(body["options"], {"include_spans": True})

        transcription = circus.select_benchmarks(self.matrix, "smoke", "transcription")[0]
        _, transcribe_body = circus.antfly_request(transcription, self.fixtures["transcription"])
        self.assertEqual(transcribe_body["model"], "openai/whisper-tiny")
        self.assertNotIn("url", transcribe_body)

        qat = next(bench for bench in self.matrix["benchmarks"] if bench["id"] == "generation-gemma4-e4b-qat")
        _, generation_body = circus.antfly_request(qat, self.fixtures["generation"])
        self.assertEqual(generation_body["draft_model"], "google/gemma-4-E4B-it-assistant")
        self.assertEqual(generation_body["speculative_k"], 2)

    def test_rc24_response_shapes(self) -> None:
        ok, _, value, _ = circus.quality_result(
            "reading / OCR",
            {"data": [{"object": "read", "index": 0, "text": "ANTFLY MOTHS"}]},
            self.fixtures["ocr"],
        )
        self.assertTrue(ok)
        self.assertEqual(value, 1.0)
        ok, _, value, _ = circus.quality_result(
            "transcription",
            {"data": [{"object": "transcription", "index": 0, "text": "antfly moths measure local inference"}]},
            self.fixtures["transcription"],
        )
        self.assertTrue(ok)
        self.assertEqual(value, 1.0)

    def test_failed_trial_stays_visible(self) -> None:
        bench = circus.select_benchmarks(self.matrix, "smoke", "generation")[0]
        row = circus.aggregate_row(bench, "antfly", [{"status": "failed", "error": "boom"}], False)
        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["error"], "boom")

    def test_provider_variants_do_not_claim_mtp_for_references(self) -> None:
        bench = next(item for item in self.matrix["benchmarks"] if item["id"] == "generation-gemma4-e4b-qat")
        trial = {"status": "ok", "latencies_ms": [1.0], "throughput_per_second": 1.0, "quality_value": 1.0}
        antfly = circus.aggregate_row(bench, "antfly", [trial], True)
        reference = circus.aggregate_row(bench, "llama.cpp", [trial], True)
        self.assertIn("MTP", antfly["variant"])
        self.assertNotIn("MTP", reference["variant"])

    def test_rc24_flat_and_legacy_model_layouts(self) -> None:
        bench = circus.select_benchmarks(self.matrix, "smoke", "dense embedding")[0]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            flat = root / bench["model"]
            flat.mkdir(parents=True)
            self.assertEqual(circus.model_dir(root, bench), flat)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy = root / bench["role"] / bench["model"]
            legacy.mkdir(parents=True)
            self.assertEqual(circus.model_dir(root, bench), legacy)

    def test_running_bundle_is_valid_and_retains_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            campaign = Path(tmp)
            result = {"schema_version": 1, "run_id": "test", "profile": "full", "category": None, "rows": [{"status": "failed"}]}
            provenance = {"schema_version": 1, "subject": {"version": "rc24", "sha256": "0" * 64}, "host": {"platform": "test", "machine": "test"}}
            receipt = {"schema_version": 1, "run_id": "test", "status": "running"}
            circus.write_campaign_bundle(campaign, result, provenance, receipt)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(circus.validate_campaign(campaign), 0)
            self.assertEqual(json.loads((campaign / "result.json").read_text())["rows"], [{"status": "failed"}])

    def test_campaign_validator_requires_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with redirect_stdout(io.StringIO()):
                self.assertEqual(circus.validate_campaign(Path(tmp)), 1)
            (Path(tmp) / "result.json").write_text(json.dumps({"schema_version": 1, "rows": []}), encoding="utf-8")
            (Path(tmp) / "provenance.json").write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
            (Path(tmp) / "receipt.json").write_text(json.dumps({"schema_version": 1, "status": "failed"}), encoding="utf-8")
            (Path(tmp) / "report.md").write_text("# report\n", encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                self.assertEqual(circus.validate_campaign(Path(tmp)), 0)


if __name__ == "__main__":
    unittest.main()
