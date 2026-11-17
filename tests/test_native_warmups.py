from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import common as C
import embed_bench as E
import gliner_bench as G


class NativeWarmupTests(unittest.TestCase):
    def test_failed_warmup_skips_measurements_but_preserves_both_healthy_siblings(self):
        for producer in (E, G):
            with self.subTest(producer=producer.__name__), tempfile.TemporaryDirectory() as temporary:
                requests = Counter()

                class Handler(BaseHTTPRequestHandler):
                    def log_message(self, *_):
                        pass

                    def do_POST(self):
                        payload = json.loads(self.rfile.read(int(self.headers["content-length"])))
                        texts = payload.get("input") or [item["content"] for item in payload["inputs"]]
                        name = texts[0]
                        requests[name] += 1
                        status = 500 if name == "broken" else 200
                        body = ({"error": {"code": "INTERNAL", "message": "warmup failed"}} if status != 200
                                else {"data": [{"embedding": [1.0, 0.0], "entities": []} for _ in texts]})
                        encoded = json.dumps(body).encode()
                        self.send_response(status)
                        self.send_header("content-type", "application/json")
                        self.send_header("content-length", str(len(encoded)))
                        self.end_headers()
                        self.wfile.write(encoded)

                http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
                thread = threading.Thread(target=http.serve_forever, daemon=True)
                thread.start()
                self.addCleanup(http.server_close)

                class Server:
                    def __init__(self, *_args, **_kwargs):
                        self.port = http.server_port

                    def start(self):
                        return {}

                    def stop(self):
                        return {}

                cells = [{"id": name, "calls": [[name], [name]], "warmup": 1,
                          "schema": {"entities": ["person"]}}
                         for name in ("early", "broken", "later")]
                args = SimpleNamespace(antfly=Path("unused-by-transport-test"), model="base")
                out = Path(temporary)
                try:
                    with patch.object(C, "AntflyServer", Server), patch.object(C, "native_precision", return_value={}):
                        result = producer.run_antfly(args, cells, out, 0)
                finally:
                    http.shutdown()
                    thread.join()
                calls = C.load_jsonl(out / "calls.jsonl")
                self.assertEqual(Counter(row["cell"] for row in calls), {"early": 2, "later": 2})
                self.assertTrue(all(row["ok"] for row in calls))
                self.assertEqual(requests, {"early": 4, "broken": 1, "later": 3})
                self.assertEqual(result["warmups"]["broken"]["state"], "failed")
                self.assertIn("error", result)


if __name__ == "__main__":
    unittest.main()
