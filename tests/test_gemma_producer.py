from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from contextlib import ExitStack, closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyze
import common as C
import gemma4_bench as G
import quality_gemma as Q
import summarize as S


class GemmaProducerTests(unittest.TestCase):
    RENDERING = "<bos>user\n  private diagnostic rendering\nassistant"
    TOKENS = [2, 190013, 990017]

    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.binary = self.root / "binary"
        self.binary.write_bytes(b"test executable identity only")
        self.requests = []
        self.rejections = {}
        self.fail_native_stop = False
        self.tokenize_status = 200
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def respond(self, status, value, stream=False):
                data = ("data: " + json.dumps(value) + "\n\ndata: [DONE]\n\n").encode() if stream else json.dumps(value).encode()
                self.send_response(status)
                self.send_header("Content-Type", "text/event-stream" if stream else "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/apply-template":
                    self.respond(200, {"prompt": owner.RENDERING})
                    return
                if self.path == "/tokenize":
                    self.respond(owner.tokenize_status, {"tokens": owner.TOKENS} if owner.tokenize_status == 200 else {"error": "tokenization unavailable"})
                    return
                text = body["messages"][0]["content"][-1]["text"]
                owner.requests.append((self.path, text))
                status = owner.rejections.get((self.path, text))
                if status == "transport":
                    self.close_connection = True
                    self.connection.close()
                    return
                if status:
                    self.respond(status, {"error": {"code": "InputLimit", "message": "request exceeds supported input"}})
                    return
                answer = "A" if "Choose one option." in text else "1"
                self.respond(200, {"choices": [{"delta": {"content": "Final answer: " + answer}, "finish_reason": "length"}],
                                   "usage": {"prompt_tokens": 3, "completion_tokens": body["max_tokens"]}}, stream=True)

        self.http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.http.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop_http)

        class Server:
            def __init__(self, *args, **kwargs):
                self.port = owner.http.server_port
                self.tag = args[-1]

            def start(self):
                return {"ready_s": 0.001, **({"version": "test reference"} if "llama" in self.tag else {})}

            def stop(self):
                if owner.fail_native_stop and "antfly" in self.tag:
                    raise RuntimeError("teardown observation unavailable")
                return {}

        for target, value in (
            ("gemma4_bench.C.AntflyServer", Server),
            ("gemma4_bench.LlamaServer", Server),
            ("gemma4_bench.C.native_precision", lambda *a, **k: {}),
            ("gemma4_bench.C.host_snapshot", lambda: {}),
            ("gemma4_bench.loading_evidence", lambda *a: {"tensor_inventory": {}, "mmap": {"observed_log_lines": []}}),
            ("gemma4_bench.C.exclusive_bench_lock", lambda: 0),
            ("gemma4_bench.C.wait_quiet", lambda *a, **k: {}),
            ("gemma4_bench.model_dir", lambda: self.root),
        ):
            self.stack.enter_context(patch(target, value))
        self.files = {"model": G.MODEL, "hf_revision": G.HF_REVISION, **{
            key: {"name": name, "sha256": digest, "size_bytes": 1, "path": self.binary}
            for key, (name, digest) in G.FILES.items()}}
        self.stack.enter_context(patch.object(G, "verify_files", return_value=self.files))
        self.args = SimpleNamespace(antfly=self.binary, llama_server=str(self.binary), backend="auto",
                                    out=str(self.root / "run"), lifecycles=2, only_cell=None, max_load=None)
        self.cells = [{"id": name, "max_tokens": 4, "warmup": 1,
                       "calls": [{"text": name + str(index)} for index in range(2)]}
                      for name in ("healthy", "blocked", "after")]

    def stop_http(self):
        self.http.shutdown()
        self.http.server_close()
        self.thread.join()

    def run_lifecycles(self):
        run = self.root / "speed" / "gemma4"
        run.mkdir(parents=True)
        with closing(C.JsonlWriter(run / "lifecycles.jsonl")) as metadata:
            for lifecycle in range(2):
                for arm in (("antfly", "llama") if lifecycle == 0 else ("llama", "antfly")):
                    info = G.run_arm(self.args, arm, self.files, self.cells, run, lifecycle)
                    metadata.write({**info, "lc": lifecycle})
        return run

    def summary(self, run):
        spec = {"workload": "gemma4", "model": G.MODEL, "model_key": G.MODEL_KEY,
                "arms": {"antfly": "antfly", "llama": "llama.cpp"},
                "cells": {cell["id"]: len(cell["calls"]) for cell in self.cells}}
        return S.speed_summary(run.parents[1], "gemma4", spec, spec["cells"], 2)

    def assert_private_payload_absent(self, value):
        if isinstance(value, dict):
            for key, child in value.items():
                self.assertNotIn(key, {"rendered_prompt", "prompt_token_ids", "stdout", "stderr"})
                self.assert_private_payload_absent(child)
        elif isinstance(value, list):
            self.assertNotEqual(value, self.TOKENS)
            for child in value:
                self.assert_private_payload_absent(child)
        self.assertNotIn(self.RENDERING, json.dumps(value, ensure_ascii=False))
        self.assertNotIn("private diagnostic rendering", json.dumps(value, ensure_ascii=False))

    def test_producer_outputs_analyze_across_multiple_fresh_lifecycles(self):
        run = self.run_lifecycles()
        rows = C.load_jsonl(run / "outputs.jsonl")
        self.assertEqual({row["lc"] for row in rows}, {0})
        self.assertEqual(len(rows), 12)
        analysis = analyze.summarize_run(run)
        healthy = analysis["cells"]["healthy"]
        self.assertEqual([row["lc"] for row in healthy["antfly"]["lifecycles"]], [0, 1])
        self.assertEqual(healthy["antfly"]["ok"], 4)
        self.assertEqual(analysis["agreement"]["antfly_vs_llama"]["healthy"]["exact"], 1)
        steps, pairs = self.summary(run)
        self.assertEqual({step["status"] for step in steps}, {"complete"})
        self.assertEqual({pair["cell"] for pair in pairs}, {"healthy", "blocked", "after"})

    def test_native_warmup_refusal_keeps_actual_calls_and_healthy_siblings(self):
        for call in self.cells[1]["calls"]:
            self.rejections[("/ai/v1/generate", call["text"])] = 413
        run = self.run_lifecycles()
        native_calls = [row for row in C.load_jsonl(run / "calls.jsonl") if row["arm"] == "antfly"]
        refused = [row for row in native_calls if row["cell"] == "blocked"]
        self.assertEqual(len(refused), 4)
        self.assertTrue(all(C.call_refused(row) for row in refused))
        self.assertTrue(all(row["warmup_state"] == "refused" for row in refused))
        self.assertEqual(sum(path == "/ai/v1/generate" and text.startswith("blocked") for path, text in self.requests), 6)
        metadata = [row for row in C.load_jsonl(run / "lifecycles.jsonl") if row["arm"] == "antfly"]
        self.assertTrue(all(row["warmups"]["blocked"]["state"] == "refused" for row in metadata))
        self.assertTrue(all(row["warmups"]["blocked"]["calls"][0]["response_error"]["code"] == "InputLimit" for row in metadata))
        steps, pairs = self.summary(run)
        native = next(step for step in steps if step["provider"] == "antfly")
        self.assertEqual(native["cells"]["blocked"]["status"], "refused")
        self.assertEqual(native["cells"]["blocked"]["refused_calls"], 4)
        self.assertEqual(set(native["metrics"]["cells"]), {"healthy", "after"})
        self.assertEqual({pair["cell"] for pair in pairs}, {"healthy", "after"})
        self.assertTrue(C.benchmark_complete(run, {"antfly", "llama"}, 2, 6))

    def assert_failed_warmup_isolated(self, status):
        self.rejections[("/ai/v1/generate", "blocked0")] = status
        run = self.run_lifecycles()
        native = [row for row in C.load_jsonl(run / "calls.jsonl") if row["arm"] == "antfly"]
        self.assertEqual({row["cell"] for row in native}, {"healthy", "after"})
        self.assertEqual(sum(path == "/ai/v1/generate" and text.startswith("blocked") for path, text in self.requests), 2)
        metadata = [row for row in C.load_jsonl(run / "lifecycles.jsonl") if row["arm"] == "antfly"]
        self.assertTrue(all(row["warmups"]["blocked"]["state"] == "failed" for row in metadata))
        steps, pairs = self.summary(run)
        native = next(step for step in steps if step["provider"] == "antfly")
        self.assertEqual(native["cells"]["blocked"]["status"], "failed")
        self.assertEqual(native["cells"]["after"]["status"], "complete")
        self.assertEqual({pair["cell"] for pair in pairs}, {"healthy", "after"})
        self.assertFalse(C.benchmark_complete(run, {"antfly", "llama"}, 2, 6))

    def test_server_warmup_failure_skips_only_that_cell_without_placeholder_calls(self):
        self.assert_failed_warmup_isolated(500)

    def test_transport_warmup_failure_skips_only_that_cell_without_placeholder_calls(self):
        self.assert_failed_warmup_isolated("transport")

    def test_later_cell_exception_and_teardown_keep_rows_and_partial_metadata(self):
        self.fail_native_stop = True
        original_stream = G.stream

        def stream(port, endpoint, payload):
            if endpoint == "/ai/v1/generate" and payload["messages"][0]["content"][-1]["text"] == "blocked1":
                raise RuntimeError("later request interrupted")
            return original_stream(port, endpoint, payload)

        with patch.object(G, "stream", stream):
            run = self.run_lifecycles()
        native = [row for row in C.load_jsonl(run / "calls.jsonl") if row["arm"] == "antfly"]
        self.assertEqual(len(native), 10)
        self.assertEqual(sum(row["cell"] == "blocked" for row in native), 2)
        self.assertEqual(sum(row["cell"] == "after" for row in native), 4)
        metadata = [row for row in C.load_jsonl(run / "lifecycles.jsonl") if row["arm"] == "antfly"]
        for row in metadata:
            self.assertIn("blocked", row["cell_errors"])
            self.assertIn("footprint", row["teardown_errors"])
            self.assertIn("weight_loading", row)
            self.assertIn("later request interrupted", row["error"])
            self.assertIn("teardown observation unavailable", row["error"])

    def test_retained_speed_and_quality_prompt_evidence_contains_only_fingerprints(self):
        diagnostic = subprocess.CompletedProcess([], 0,
            stdout=f"prompt:\n{self.RENDERING}\nprompt_token_ids: {' '.join(map(str, self.TOKENS))}\n",
            stderr="diagnostic startup log")
        with patch.object(G.subprocess, "run", return_value=diagnostic), patch.object(G, "make_cells", return_value=self.cells):
            self.assertTrue(G.orchestrate(self.args))
            speed = Path(self.args.out)
            native = json.loads((speed / "native_prompt_evidence.json").read_text())
            reference = json.loads((speed / "llama_prompt_evidence.json").read_text())
            self.assert_private_payload_absent(native)
            self.assert_private_payload_absent(reference)
            fingerprint = native[0]["native"]
            self.assertEqual(fingerprint["rendered_prompt_sha256"], hashlib.sha256(self.RENDERING.encode()).hexdigest())
            self.assertEqual(fingerprint["rendered_prompt_bytes"], len(self.RENDERING.encode()))
            self.assertEqual(fingerprint["prompt_token_ids_sha256"], hashlib.sha256(json.dumps(self.TOKENS, separators=(",", ":")).encode()).hexdigest())
            self.assertEqual(fingerprint["prompt_token_count"], 3)
            comparison = json.loads((speed / "prompt_comparisons.json").read_text())[0]
            self.assertTrue(comparison["diagnostic_token_ids_equal"])
            self.assertTrue(comparison["diagnostic_rendering_equal"])
            self.assertEqual(comparison["http_generation_prompt_parity"], "unproven")
            tasks = [{"dataset": dataset, "id": str(index), "question": "task question " + str(index), "answer": answer,
                      **({"choices": {"label": ["A", "B"], "text": ["one", "two"]}} if dataset == "arc_challenge" else {})}
                     for dataset, answer in (("gsm8k", "1"), ("arc_challenge", "A")) for index in range(12)]
            self.args.out = str(self.root / "quality" / "gemma4")
            with patch.object(Q, "check", return_value=(tasks, {})):
                result = Q.run(self.args)
            retained = json.loads((Path(self.args.out) / "result.json").read_text())
            self.assert_private_payload_absent(retained)
            canonical_steps, canonical_pairs = S.quality_summary(self.root, "gemma4", {
                "workload": "gemma4", "model": G.MODEL, "arms": {"antfly": "antfly", "llama": "llama.cpp"}})
            self.assert_private_payload_absent({"steps": canonical_steps, "comparisons": canonical_pairs})
            self.assertEqual({step["status"] for step in canonical_steps}, {"complete"})
            self.assertEqual(result["comparisons"]["gsm8k"]["scores"]["antfly"]["correct"], 12)
            self.assertTrue(result["prompt_comparisons"]["gsm8k"]["diagnostic_token_ids_equal"])
            for provider in retained["providers"].values():
                self.assertEqual(provider["prompt_diagnostics"]["gsm8k"]["prompt_token_count"], 3)
                self.assertEqual(provider["prompt_diagnostics"]["gsm8k"]["rendered_prompt_sha256"], fingerprint["rendered_prompt_sha256"])

    def test_tokenizer_failure_does_not_retain_the_successfully_rendered_prompt(self):
        self.tokenize_status = 500
        payload = G.request_body({"text": "diagnostic task"}, {"max_tokens": 4}, "llama")
        evidence = G.llama_prompt_evidence(self.http.server_port, payload)
        self.assertEqual(evidence["status"], "unavailable")
        self.assertEqual(evidence["rendered_prompt_sha256"], hashlib.sha256(self.RENDERING.encode()).hexdigest())
        self.assertNotIn("prompt_token_ids_sha256", evidence)
        self.assert_private_payload_absent(evidence)

    def test_cli_timeout_does_not_retain_command_prompt_or_stdout(self):
        secret = "private diagnostic rendering"
        failure = subprocess.TimeoutExpired([str(self.binary), secret], 900, output=secret)
        with patch.object(G.subprocess, "run", side_effect=failure):
            evidence = G.native_prompt_evidence(self.binary, {"text": secret})
        self.assertEqual(evidence["status"], "unavailable")
        self.assert_private_payload_absent(evidence)


if __name__ == "__main__":
    unittest.main()
