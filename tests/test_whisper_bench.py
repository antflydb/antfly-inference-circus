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

    def test_short_fixture_rejects_changed_bytes_and_invalid_format(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "short.wav"
            manifest = root / "media_manifest.json"

            def write_fixture(channels: int) -> None:
                with wave.open(str(path), "wb") as output:
                    output.setnchannels(channels)
                    output.setsampwidth(2)
                    output.setframerate(16000)
                    output.writeframes(b"\0\0" * channels * 160)
                manifest.write_text(json.dumps({"librispeech": {
                    "revision": "5be91486e11a2d616f4ec5db8d3fd248585ac07a"},
                    "audio": [{"name": path.name, "id": "clip", "seconds": 0.01,
                               "text": "HELLO WORLD",
                               "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}]}))

            with patch.object(whisper, "MEDIA_MANIFEST", manifest), patch.object(whisper.C, "MEDIA", root):
                write_fixture(1)
                fixture = whisper.make_cells("short")[0]["calls"][0]
                self.assertEqual(fixture["path"], path)
                self.assertEqual(fixture["ref"], "HELLO WORLD")
                path.write_bytes(path.read_bytes() + b"changed")
                with self.assertRaisesRegex(ValueError, "checksum differs"):
                    whisper.make_cells("short")
                write_fixture(2)
                with self.assertRaisesRegex(ValueError, "unexpected WAV format"):
                    whisper.make_cells("short")

    def test_removed_cells_are_rejected_by_reference_child(self) -> None:
        for cell in ("batch4", "multiwindow60"):
            argv = ["pyarm", "--backend", "auto", "--out", "/unused", "--lc", "0",
                    "--only-cell", cell]
            with self.subTest(cell=cell):
                with patch("sys.stderr"), self.assertRaises(SystemExit) as raised:
                    whisper.main(argv)
                self.assertEqual(raised.exception.code, 2)

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
                     "converter": {"name": "whisper.cpp", "revision": whisper.CPP_REVISION},
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
