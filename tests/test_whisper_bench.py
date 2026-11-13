from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("whisper_bench_test_module", ROOT / "scripts" / "whisper_bench.py")
whisper = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(whisper)


class WhisperBenchTests(unittest.TestCase):
    def test_normalization_wer_and_exactness(self) -> None:
        self.assertEqual(whisper.normalize_text(" Hello, WORLD! "), "hello world")
        metric = whisper.quality(["Hello brave world"], ["HELLO WORLD"])
        self.assertEqual(metric["word_edits"], 1)
        self.assertEqual(metric["reference_words"], 2)
        self.assertEqual(metric["wer"], 0.5)
        self.assertEqual(metric["exact_normalized_count"], 0)

    def test_cells_cover_short_batch_and_real_multiwindow_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            media = root / "media"
            media.mkdir()
            entries = []
            for index in range(8):
                path = media / f"aud{index:02}.wav"
                with wave.open(str(path), "wb") as output:
                    output.setnchannels(1)
                    output.setsampwidth(2)
                    output.setframerate(16000)
                    output.writeframes(b"\0\0" * 16000 * 8)
                entries.append({"name": path.name, "id": f"clip-{index}", "seconds": 8.0,
                                "text": f"CLIP NUMBER {index}",
                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            manifest = root / "media_manifest.json"
            manifest.write_text(json.dumps({"librispeech": {
                "revision": "5be91486e11a2d616f4ec5db8d3fd248585ac07a"}, "audio": entries}))
            with patch.object(whisper, "MEDIA_MANIFEST", manifest), patch.object(whisper.C, "MEDIA", media):
                cells = whisper.make_cells(None, root / "out")
            self.assertEqual([cell["id"] for cell in cells], whisper.CELLS)
            self.assertEqual(len(cells[0]["calls"]), 1)
            self.assertEqual(len(cells[1]["calls"]), 4)
            self.assertEqual(len(cells[2]["calls"]), 1)
            self.assertGreaterEqual(cells[2]["calls"][0]["seconds"], 60)
            self.assertEqual(cells[2]["calls"][0]["ref"], " ".join(e["text"] for e in entries))

    def test_cpp_native_timing_requires_valid_load_and_total_fields(self) -> None:
        parsed = whisper._parse_cpp_timings(
            "whisper_print_timings: load time = 125.25 ms\n"
            "whisper_print_timings: total time = 900.50 ms\n")
        self.assertEqual(whisper._cpp_compute_ms(parsed), 775.25)
        with self.assertRaises(ValueError):
            whisper._cpp_compute_ms({"total_time": 900.5})
        with self.assertRaises(ValueError):
            whisper._cpp_compute_ms({"load_time": 900.5, "total_time": 900.5})

    def test_cpp_provenance_is_bound_to_source_and_converted_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "ggml-small.bin"
            model.write_bytes(b"converted whisper fixture")
            converted_sha = hashlib.sha256(model.read_bytes()).hexdigest()
            provenance = root / "provenance.json"
            value = {"schema_version": 1, "model": whisper.MODEL,
                     "source_revision": whisper.REVISION,
                     "source_model_sha256": whisper.SOURCE_WEIGHT_SHA256,
                     "converted_model_sha256": converted_sha,
                     "converter": {"name": "whisper.cpp", "revision": "a" * 40},
                     "private_path": "/must/not/be-published"}
            provenance.write_text(json.dumps(value))
            public = whisper.validate_cpp_provenance(model, provenance)
            self.assertEqual(public["converted_model_sha256"], converted_sha)
            self.assertNotIn("private_path", public)
            value["source_revision"] = "wrong"
            provenance.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                whisper.validate_cpp_provenance(model, provenance)

    def test_agreement_requires_every_engine_and_reports_pairwise_wer(self) -> None:
        cell = {"id": "short", "calls": [{"ref": "hello world"}]}
        results = {
            "antfly": {"rows": [{"call": {"cell": "short"}, "texts": ["Hello, world!"]}]},
            "transformers": {"rows": [{"call": {"cell": "short"}, "texts": ["hello world"]}]},
            "whisper.cpp": {"rows": [{"call": {"cell": "short"}, "texts": ["HELLO WORLD"]}]},
        }
        agreement = whisper.agreement_rows(0, [cell], results)[0]
        self.assertTrue(agreement["complete"])
        self.assertTrue(agreement["all_normalized_exact"])
        self.assertEqual(agreement["normalized_exact_count"], 1)
        self.assertEqual(agreement["raw_exact_count"], 0)
        self.assertTrue(all(value == 0 for value in agreement["pairwise"].values()))
        del results["whisper.cpp"]
        self.assertFalse(whisper.agreement_rows(0, [cell], results)[0]["complete"])


if __name__ == "__main__":
    unittest.main()
