from __future__ import annotations

import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch
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
    def test_quality_negative_is_structural_and_exact(self) -> None:
        negative = circus.quality_terminal_negative(
            "relevant cosine margin", -0.25, "cosine"
        )
        self.assertEqual(
            set(negative),
            {"kind", "stage", "reason", "observed"},
        )
        self.assertEqual(negative["kind"], "quality-gate")
        self.assertEqual(negative["reason"], "relevant cosine margin")
        self.assertEqual(negative["observed"]["quality_value"], -0.25)

        provider_path = ROOT / "providers" / "reference.py"
        provider_spec = importlib.util.spec_from_file_location(
            "termite_reference_provider_test", provider_path
        )
        assert provider_spec is not None and provider_spec.loader is not None
        provider = importlib.util.module_from_spec(provider_spec)
        provider_spec.loader.exec_module(provider)
        result = provider.quality_failure(
            [1.0, 2.0],
            "expected document ranked first",
            "document 2 ranked first",
            0.0,
            "pass",
            {"scores": [0.1, 0.2, 0.9]},
        )
        self.assertEqual(result["status"], "failed")
        self.assertEqual(
            set(result["terminal_negative"]),
            {"kind", "stage", "reason", "observed"},
        )
        self.assertEqual(result["latencies_ms"], [1.0, 2.0])
        self.assertIn("response", result)

    def test_mixed_or_failed_trials_never_publish_performance(self) -> None:
        bench = circus.select_benchmarks(self.matrix, "smoke", "generation")[0]
        passed = {
            "status": "ok",
            "latencies_ms": [1.0],
            "throughput_per_second": 1000.0,
            "quality_gate": "nonempty completion",
            "quality_value": 1.0,
            "quality_unit": "pass",
            "memory": {"peak_memory_mb": 100.0},
        }
        failed = {
            "status": "failed",
            "error": "quality gate failed: nonempty completion",
            "latencies_ms": [2.0],
            "throughput_per_second": 500.0,
            "quality_gate": "nonempty completion",
            "quality_value": 0.0,
            "quality_unit": "pass",
            "memory": {"peak_memory_mb": 200.0},
            "terminal_negative": circus.quality_terminal_negative(
                "nonempty completion", 0.0, "pass"
            ),
        }
        for trials in ([passed, failed], [failed, failed]):
            row = circus.aggregate_row(bench, "antfly", list(trials), True)
            self.assertEqual(row["status"], "failed")
            for field in (
                "latency_p50_ms",
                "latency_p95_ms",
                "throughput_per_second",
                "peak_memory_mean_mb",
                "peak_memory_range_mb",
            ):
                self.assertIsNone(row[field])
            self.assertFalse(row["memory_publishable"])


    def test_cache_adapter_promotes_only_complete_quality_negatives(self) -> None:
        adapter_path = (
            ROOT.parent
            / "adapters"
            / "inference"
            / "termite-circus"
            / "cached_circus.py"
        )
        adapter_spec = importlib.util.spec_from_file_location(
            "termite_cached_circus_test", adapter_path
        )
        assert adapter_spec is not None and adapter_spec.loader is not None
        adapter = importlib.util.module_from_spec(adapter_spec)
        adapter_spec.loader.exec_module(adapter)

        class Module:
            @staticmethod
            def write_json(path: Path, value: dict) -> None:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value), encoding="utf-8")

        class Cache:
            promoted: list[tuple[Path, dict]] = []

            def promote(self, source: Path, desired: dict) -> None:
                self.promoted.append((source, desired))

        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            "os.environ", {"QUIET_HOST_ACK": "1"}
        ):
            campaign = Path(tmp)
            raw = campaign / "raw"
            raw.mkdir()
            (raw / "trial.log").write_text("fixture\n", encoding="utf-8")
            (raw / "trial-memory.jsonl").write_text(
                '{"sample":1}\n', encoding="utf-8"
            )
            terminal_negative = circus.quality_terminal_negative(
                "fixture gate", 0.0, "pass"
            )
            result = {
                "status": "failed",
                "error": "quality gate failed: fixture gate",
                "latencies_ms": [1.0, 2.0],
                "throughput_per_second": 500.0,
                "quality_gate": "fixture gate",
                "quality_value": 0.0,
                "quality_unit": "pass",
                "response": {"value": "wrong"},
                "terminal_negative": terminal_negative,
                "memory": {
                    "sample_count": 2,
                    "timeline": "raw/trial-memory.jsonl",
                },
                "raw_log": "raw/trial.log",
            }
            cell = {
                "desired": {
                    "protocol_identity": {"iterations_per_trial": 2},
                    "required_roles": [
                        "normalized-result",
                        "raw",
                        "memory-timeline",
                        "log",
                        "quiet-host",
                    ],
                }
            }
            cache = Cache()
            adapter.promote_trial(
                Module,
                cache,
                cell,
                raw,
                "antfly",
                "fixture",
                1,
                result,
                "2026-08-01T00:00:00Z",
                "2026-08-01T00:01:00Z",
            )
            self.assertEqual(len(cache.promoted), 1)
            promoted = cache.promoted[0][1]
            self.assertEqual(promoted["evidence"]["status"], "failed")
            self.assertEqual(
                promoted["evidence"]["terminal_negative"], terminal_negative
            )

            execution = dict(result)
            execution.pop("terminal_negative")
            execution["error"] = "provider timed out"
            adapter.promote_trial(
                Module,
                cache,
                cell,
                raw,
                "antfly",
                "fixture",
                2,
                execution,
                "2026-08-01T00:02:00Z",
                "2026-08-01T00:03:00Z",
            )
            self.assertEqual(len(cache.promoted), 1)


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
