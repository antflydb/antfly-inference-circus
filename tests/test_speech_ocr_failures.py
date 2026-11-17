from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name + "_failure_tests", ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


W = load("whisper_bench")
O = load("ocr_bench")


@contextmanager
def endpoint(respond):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            requests.append((self.path, body))
            status, payload = respond(len(requests), self.path, body)
            if status is None:
                self.close_connection = True
                return
            self.send_response(status)
            self.end_headers()
            self.wfile.write(payload if isinstance(payload, bytes) else json.dumps(payload).encode())

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port, requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def success(path, body):
    if path == "/ai/v1/read":
        return {"data": [{"text": "hello world"} for _ in json.loads(body)["images"]]}
    return {"text": "hello world"}


class SpeechOcrFailureTests(unittest.TestCase):
    @contextmanager
    def native(self, module, root, port, *, cleanup_error=False):
        class Server:
            def __init__(self, *_args, **_kwargs):
                self.port = port

            def start(self, **_kwargs):
                return {"server_pid": 123}

            def stop(self):
                if cleanup_error:
                    raise RuntimeError("cleanup failed")
                return {}

        with ExitStack() as stack:
            stack.enter_context(patch.object(module.C, "AntflyServer", Server))
            stack.enter_context(patch.object(module.C, "file_identity", return_value={}))
            stack.enter_context(patch.object(module.C, "native_precision", return_value={}))
            stack.enter_context(patch.object(module.C, "host_snapshot", return_value={}))
            yield argparse.Namespace(antfly=root / "antfly", repetitions=2)

    def cells(self, module, root, count=3):
        names = ["short", "long", "medium"] if module is W else ["single_page", "batch4_pages", "healthy_page"]
        cells = []
        for name in names[:count]:
            path = root / name
            path.write_bytes(name.encode())
            cells.append({"id": name, "calls": [{"id": name + str(index), "path": path,
                "ref": "hello world", "seconds": 1.0} for index in range(
                    4 if module is O and name == "batch4_pages" else 1)]})
        return cells

    def test_warmup_refusal_is_not_replayed_and_hard_failure_skips_only_its_cell(self):
        cases = [(400, {"error": {"message": "unsupported request"}}, True),
                 (413, {"error": {"message": "request too large"}}, True),
                 (422, {"error": {"message": "unsupported shape"}}, True),
                 (500, {"error": {"message": "server failed"}}, False),
                 (400, b"not JSON", False),
                 (None, b"", False)]
        for module in (W, O):
            for status, payload, refused in cases:
                with self.subTest(workload=module.WORKLOAD, status=status, refused=refused), tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    cells = self.cells(module, root)
                    target_audio = json.loads(W._antfly_request(cells[1]["calls"][0]))["audio"] if module is W else None

                    def respond(number, path, body):
                        target = (json.loads(body)["audio"] == target_audio if module is W
                                  else len(json.loads(body)["images"]) == 4)
                        return (status, payload) if target else (200, success(path, body))

                    with endpoint(respond) as (port, requests), self.native(module, root, port) as args:
                        result = module.run_antfly(args, cells, root, 7)
                    module._write_arm(root, result, 7)
                    calls = module.C.load_jsonl(root / "calls.jsonl")
                    self.assertEqual([call["cell"] for call in calls if call["ok"]],
                                     [cells[0]["id"]] * 2 + [cells[2]["id"]] * 2)
                    self.assertTrue(all(call["quality"]["wer"] == 0 for call in calls if call["ok"]))
                    warmups = result["warmup"]["cells"] if module is W else result["warmups"]
                    self.assertEqual(warmups[1]["state"], "refused" if refused else "failed")
                    self.assertNotIn("ms", warmups[1])
                    if refused:
                        self.assertNotIn("error", result)
                        refused_calls = [call for call in calls if module.C.call_refused(call)]
                        self.assertEqual([call["repetition"] for call in refused_calls], [0, 1])
                        self.assertTrue(all(call["warmup_state"] == "refused" and "ms" not in call for call in refused_calls))
                        self.assertEqual(len(requests), 9)
                    else:
                        self.assertIn("error", result)
                        self.assertEqual(len(calls), 4)
                        self.assertEqual(len(requests), 7)

    def test_measured_failure_and_cleanup_preserve_completed_rows_and_later_siblings(self):
        for module in (W, O):
            with self.subTest(workload=module.WORKLOAD), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                cells = self.cells(module, root)

                def respond(number, path, body):
                    if number == 5:
                        return 500, {"error": {"message": "measured failure"}}
                    return 200, success(path, body)

                with endpoint(respond) as (port, requests), self.native(module, root, port, cleanup_error=True) as args:
                    result = module.run_antfly(args, cells, root, 3)
                module._write_arm(root, result, 3)
                calls = module.C.load_jsonl(root / "calls.jsonl")
                self.assertEqual([(call["cell"], call["repetition"], call["ok"]) for call in calls],
                    [(cells[0]["id"], 0, True), (cells[0]["id"], 1, True),
                     (cells[1]["id"], 0, False), (cells[2]["id"], 0, True), (cells[2]["id"], 1, True)])
                failed = calls[2]
                self.assertFalse(module.C.call_refused(failed))
                self.assertEqual(failed["status"], 500)
                self.assertEqual(failed["input_ids"], [item["id"] for item in cells[1]["calls"]])
                self.assertNotIn("ms", failed)
                self.assertEqual({error["stage"] for error in result["errors"]}, {"measured", "cleanup"})
                self.assertEqual(len(requests), 8)
                self.assertEqual(len(module.C.load_jsonl(root / "outputs.jsonl")), 4)

    def test_cpp_http_status_is_structured_but_500_and_malformed_400_are_not_refusals(self):
        for status, payload, refused in [(422, {"error": {"message": "unsupported"}}, True),
                                         (500, {"error": {"message": "failed"}}, False),
                                         (400, b"invalid", False)]:
            with self.subTest(status=status, refused=refused), endpoint(lambda *_: (status, payload)) as (port, requests):
                with self.assertRaises(W.C.HttpResponseError) as raised:
                    W._cpp_one(port, (b"audio bytes", "application/octet-stream"))
                self.assertEqual(W.C.call_refused(raised.exception.call_fields()), refused)
                self.assertEqual(len(requests), 1)


if __name__ == "__main__":
    unittest.main()
