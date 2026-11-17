from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import textwrap
import unittest
import wave
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import whisper_bench as W
import quality_whisper as Q


class WhisperEvidenceTests(unittest.TestCase):
    def test_micro_wer_cer_use_reference_denominators(self):
        score = W.quality(["", "ONE TWO THREE FOUR FIVE SIX SEVEN EIGHT NINE TEN"],
                          ["ONE", "ONE TWO THREE FOUR FIVE SIX SEVEN EIGHT NINE TEN"])
        self.assertEqual(score["word_edits"], 1)
        self.assertEqual(score["reference_words"], 11)
        self.assertEqual(score["wer"], 1 / 11)
        self.assertEqual(score["character_edits"], 3)
        self.assertEqual(score["cer"], 3 / score["reference_characters"])
        self.assertEqual(score["clips"], 2)
        self.assertEqual(score["exact_normalized_count"], 1)

    def test_normalization_preserves_contractions_and_expands_nfkc(self):
        score = W.quality(["ＴＨＡＴ'S a CAT!"], ["That's a cat."])
        self.assertEqual(score["wer"], 0)
        self.assertEqual(score["cer"], 0)
        self.assertEqual(score["reference_words"], 3)
        self.assertEqual(score["reference_characters"], 10)
        with self.assertRaises(ValueError):
            W.quality([], [])
        with self.assertRaises(ValueError):
            W.quality(["hello"], [])
        with self.assertRaises(ValueError):
            W.quality(["hello"], ["..."])

    def test_request_and_audio_rate_units_are_not_interchanged(self):
        cell = {"id": "short", "calls": [{"ref": "hello", "seconds": 12.0}]}
        row = W._cell_row("antfly", cell, ["hello"], 2000.0, 0)
        self.assertEqual(row["requests_per_second"], 0.5)
        self.assertEqual(row["audio_x_realtime"], 6.0)
        self.assertEqual(row["audio_real_time_factor"], 1 / 6)
        self.assertEqual(row["n_inputs"], 1)
        for duration in (0, -1, float("nan"), float("inf")):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                W._cell_row("antfly", cell, ["hello"], duration, 0)

    def test_short_audio_metrics_use_exact_samples_not_rounded_manifest_duration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "short.wav"
            with wave.open(str(audio), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16000)
                output.writeframes(b"\0\0" * 124)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "librispeech": {"revision": Q.REVISION},
                "audio": [{"id": "duration", "name": audio.name, "seconds": 0.01,
                           "text": "hello", "sha256": hashlib.sha256(audio.read_bytes()).hexdigest()}],
            }))
            with patch.object(W, "MEDIA_MANIFEST", manifest), patch.object(W.C, "MEDIA", root):
                cell = W.make_cells("short")[0]
            self.assertEqual(cell["calls"][0]["seconds"], 124 / 16000)
            row = W._cell_row("antfly", cell, ["hello"], 1000, 0)
            self.assertEqual(row["audio_x_realtime"], 124 / 16000)

    def test_agreement_includes_later_repetitions_and_marks_missing_operation_incomplete(self):
        cell = {"id": "short", "calls": [{"ref": "hello"}]}
        results = {provider: {"rows": [{"call": {"cell": "short", "repetition": repetition}, "texts": [text]}
                    for repetition, text in enumerate(["hello", "hello"])]} for provider in W.PROVIDERS}
        results["whisper.cpp"]["rows"][1]["texts"] = ["different"]
        agreement = W.agreement_rows(0, [cell], results)[0]
        self.assertEqual(agreement["items"], 2)
        self.assertEqual(agreement["normalized_exact_count"], 1)
        self.assertFalse(agreement["all_normalized_exact"])
        results["whisper.cpp"]["rows"].pop()
        incomplete = W.agreement_rows(0, [cell], results)[0]
        self.assertFalse(incomplete["complete"])
        self.assertNotIn("normalized_exact_count", incomplete)
        results["whisper.cpp"]["rows"].append(
            {"call": {"cell": "short", "repetition": 2}, "texts": ["hello"]})
        self.assertFalse(W.agreement_rows(0, [cell], results)[0]["complete"])

    def test_quality_scoring_rejects_missing_duplicate_and_reordered_clips(self):
        manifest = {"audio": [{"id": "a", "text": "one two"}, {"id": "b", "text": "three"}]}
        def arm(ids):
            return {"rows": [{"call": {"cell": id_}, "texts": ["one"]} for id_ in ids]}
        for ids in (["a"], ["a", "a"], ["b", "a"]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                Q.score_provider(arm(ids), manifest)
        scored = Q.score_provider(arm(["a", "b"]), manifest)
        self.assertEqual(scored["scores"]["word_edits"], 2)
        self.assertEqual(scored["scores"]["reference_words"], 3)

    def test_paired_difference_retains_direction_and_denominators(self):
        left = {"status": "complete", "scores": W.quality(["one"], ["one two"])}
        right = {"status": "complete", "scores": W.quality(["one two"], ["one two"])}
        comparison = Q.comparisons({"antfly": left, "transformers": right})[0]
        self.assertEqual(comparison["wer_difference"], 0.5)
        self.assertEqual(comparison["reference_words"], 2)
        self.assertEqual(comparison["paired_word_edit_differences"], [1])
        right["scores"] = W.quality(["one"], ["one"])
        with self.assertRaisesRegex(ValueError, "denominators differ"):
            Q.comparisons({"antfly": left, "transformers": right})

    def test_persistent_process_warms_each_cell_and_records_independent_requests(self):
        # A real local subprocess/HTTP endpoint exercises the lifecycle boundary,
        # independent requests, warmup exclusion and process cleanup without models.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable = root / "whisper-server"
            executable.write_text(f"#!{sys.executable}\n" + textwrap.dedent('''\
                from http.server import BaseHTTPRequestHandler, HTTPServer
                from pathlib import Path
                import json, sys
                args = sys.argv
                model = Path(args[args.index('--model') + 1])
                loads = model.with_suffix('.loads')
                loads.write_text(str(int(loads.read_text()) + 1) if loads.exists() else '1')
                class Handler(BaseHTTPRequestHandler):
                    operations = 0
                    def log_message(self, *args): pass
                    def do_GET(self):
                        self.send_response(200)
                        self.end_headers()
                        self.wfile.write(b'{"status":"ok"}')
                    def do_POST(self):
                        self.rfile.read(int(self.headers['Content-Length']))
                        Handler.operations += 1
                        text = 'warmup' if Handler.operations in (1, 5) else 'operation' + str(Handler.operations)
                        self.send_response(200)
                        self.end_headers()
                        self.wfile.write(json.dumps({'text': text}).encode())
                HTTPServer(('127.0.0.1', int(args[args.index('--port') + 1])), Handler).serve_forever()
                '''), encoding="utf-8")
            executable.chmod(0o755)
            model = root / "model.bin"
            model.write_bytes(struct.pack("<12i", 0x67676D6C, *([1] * 11)))
            provenance = root / "provenance.json"
            provenance.write_text(json.dumps({"schema_version": 1, "model": W.MODEL,
                "source_revision": W.REVISION, "source_model_sha256": W.SOURCE_WEIGHT_SHA256,
                "converted_model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                "converter": {"name": "whisper.cpp", "revision": W.CPP_REVISION}}))
            audio = root / "clip.wav"
            audio.write_bytes(b"lifecycle test audio")
            fixture_cells = [
                {"id": "short", "calls": [{"path": audio, "ref": "operation", "seconds": 5.86}]},
                {"id": "long", "calls": [{"path": audio, "ref": "operation", "seconds": 29.4}]},
            ]
            class NoSampler:
                def __init__(self, *args): pass
                def stop(self): return {}
            with patch.dict(os.environ, {"TERMITE_WHISPER_CPP": str(executable),
                    "TERMITE_WHISPER_CPP_MODEL": str(model), "TERMITE_WHISPER_CPP_PROVENANCE": str(provenance)}), \
                    patch.object(W.C, "Sampler", NoSampler):
                result = W.run_cpp_arm(fixture_cells, root, 0, repetitions=3)
            self.assertEqual(model.with_suffix(".loads").read_text(), "1")
            self.assertEqual(result["warmup"]["inference_count"], 2)
            self.assertEqual([entry["cell"] for entry in result["warmup"]["cells"]], ["short", "long"])
            self.assertEqual([row["texts"] for row in result["rows"]],
                             [["operation2"], ["operation3"], ["operation4"],
                              ["operation6"], ["operation7"], ["operation8"]])
            self.assertEqual([row["call"]["repetition"] for row in result["rows"]], [0, 1, 2, 0, 1, 2])
            self.assertEqual([row["call"]["cell"] for row in result["rows"]], ["short"] * 3 + ["long"] * 3)
            for row in result["rows"]:
                self.assertGreater(row["call"]["ms"], 0)
                self.assertEqual(row["call"]["n_inputs"], 1)
            with self.assertRaises(ProcessLookupError):
                os.kill(result["server_pid"], 0)


if __name__ == "__main__":
    unittest.main()
